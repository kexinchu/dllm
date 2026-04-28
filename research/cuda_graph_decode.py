"""CUDA-graph-accelerated greedy decode with SRP-FP32 patches.

Implementation strategy:
  D.1 Use HF `StaticCache` so the KV cache is a fixed-shape tensor that can be
      updated in-place by the model's forward pass (cudagraph-friendly).
  D.2 Pre-allocate `input_ids_buf` and `cache_position_buf` on GPU; decode
      writes into these buffers in-place each step.
  D.3 Capture one decode step into `torch.cuda.CUDAGraph()`. Replay it for
      every subsequent step. All 308+ Triton/cuBLAS kernels in the step run
      as one graph launch, eliminating per-kernel CPU dispatch.

Caveats:
  * One graph per (model, bs). Different bs ⇒ different `input_ids_buf`
    shape ⇒ different graph. We capture lazily on first call.
  * Model's `forward` must be deterministic w.r.t. inputs of fixed shape.
    Our SRP-FP32 patches satisfy this (Triton kernels are pure functions of
    pointers + sizes, and StaticCache uses indexed in-place writes).
  * Argmax of last-position logits stays outside the captured graph (it's
    cheap and lets us write the next token into `input_ids_buf` for the
    next replay).
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache


# ─────────────────────────────────────────────────────────────────────────────
# Cudagraph-accelerated decoder
# ─────────────────────────────────────────────────────────────────────────────
class CudaGraphDecoder:
    """Wraps a HuggingFace causal LM for graph-replayed greedy decoding.

    Usage:
        dec = CudaGraphDecoder(model, max_seq_len=1024)
        tokens, probs = dec.greedy(prompt_ids, max_new=256)
    """
    def __init__(self, model, max_seq_len: int = 1024):
        self.model = model
        self.config = model.config
        self.max_seq_len = max_seq_len
        self.device = next(model.parameters()).device
        self.dtype  = next(model.parameters()).dtype

        # Per-bs cached state
        self._caches = {}    # bs -> (StaticCache, input_buf, position_buf, graph, logits_out)

    # ── Setup state for a given bs ───────────────────────────────────────────
    def _prepare(self, bs: int):
        if bs in self._caches:
            return self._caches[bs]

        kv_cache = StaticCache(
            config=self.config,
            max_batch_size=bs,
            max_cache_len=self.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        input_buf    = torch.zeros((bs, 1), dtype=torch.long, device=self.device)
        position_buf = torch.zeros((1,),    dtype=torch.long, device=self.device)
        # 2-D attention_mask of shape (bs, max_seq_len). HF auto-constructs
        # the 4-D causal mask from this each call. Values: 1 for valid
        # positions (filled), 0 for invalid (unfilled). Updated in-place.
        attn_mask_buf = torch.zeros((bs, self.max_seq_len), dtype=torch.long,
                                    device=self.device)
        self._caches[bs] = [kv_cache, input_buf, position_buf, attn_mask_buf,
                            None, None]
        return self._caches[bs]

    # ── Capture one decode step into a cudagraph ─────────────────────────────
    def _capture_decode_step(self, bs: int, current_pos: int):
        state = self._caches[bs]
        kv_cache, input_buf, position_buf, attn_mask_buf, _g, _l = state

        position_buf.fill_(current_pos)
        # Mark valid positions in 2D mask
        attn_mask_buf.zero_()
        attn_mask_buf[:, :current_pos + 1] = 1

        # Warmup runs on a side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    self.model(
                        input_ids=input_buf,
                        attention_mask=attn_mask_buf,
                        cache_position=position_buf,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                out = self.model(
                    input_ids=input_buf,
                    attention_mask=attn_mask_buf,
                    cache_position=position_buf,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            logits_out = out.logits

        state[4] = graph
        state[5] = logits_out

    # ── Greedy decode ────────────────────────────────────────────────────────
    @torch.no_grad()
    def greedy(self, prompt_ids: torch.Tensor, max_new: int, use_graph: bool = True):
        """prompt_ids: [bs, L_in] long tensor on device.
        Returns (tokens_row0[max_new], probs_row0[max_new]).
        If use_graph=False, runs the decode loop without cudagraph capture
        for debugging: same kernel set, same data, but eager dispatch."""
        bs, L_in = prompt_ids.shape
        assert L_in + max_new <= self.max_seq_len, \
            f"L_in + max_new = {L_in + max_new} > max_seq_len {self.max_seq_len}"

        state = self._prepare(bs)
        kv_cache, input_buf, position_buf, attn_mask_buf, _g, _l = state

        kv_cache.reset()
        state[4] = None  # invalidate any prior graph
        state[5] = None
        attn_mask_buf.zero_()
        attn_mask_buf[:, :L_in] = 1

        # ── Prefill: feed entire prompt at positions 0..L_in-1 ──
        prefill_position = torch.arange(L_in, device=self.device, dtype=torch.long)
        out = self.model(
            input_ids=prompt_ids,
            attention_mask=attn_mask_buf,
            cache_position=prefill_position,
            past_key_values=kv_cache,
            use_cache=True,
        )
        last_logits = out.logits[:, -1]
        p = F.softmax(last_logits.float(), dim=-1)
        max_p, max_t = p.max(dim=-1)
        tokens = [int(max_t[0].item())]
        probs  = [float(max_p[0].item())]

        input_buf[:, 0] = max_t

        if use_graph:
            self._capture_decode_step(bs, current_pos=L_in)
            graph, logits_out = state[4], state[5]
            for step in range(1, max_new):
                cur_pos = L_in + step
                position_buf.fill_(cur_pos)
                attn_mask_buf[:, cur_pos] = 1  # reveal new position
                graph.replay()
                last_logits = logits_out[:, -1]
                p = F.softmax(last_logits.float(), dim=-1)
                max_p, max_t = p.max(dim=-1)
                tokens.append(int(max_t[0].item()))
                probs.append(float(max_p[0].item()))
                input_buf[:, 0] = max_t
        else:
            # eager-dispatch decode loop, no graph capture (debug)
            for step in range(1, max_new):
                cur_pos = L_in + step
                position_buf.fill_(cur_pos)
                attn_mask_buf[:, cur_pos] = 1
                out = self.model(
                    input_ids=input_buf,
                    attention_mask=attn_mask_buf,
                    cache_position=position_buf,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                last_logits = out.logits[:, -1]
                p = F.softmax(last_logits.float(), dim=-1)
                max_p, max_t = p.max(dim=-1)
                tokens.append(int(max_t[0].item()))
                probs.append(float(max_p[0].item()))
                input_buf[:, 0] = max_t

        return tokens, probs


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
def _selftest():
    from research.methods import method_BF16, method_SRP_FP32, ALL_SITES

    MODEL = '/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B'
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print('Loading model...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()

    prompt = "Solve the equation 3x + 5 = 14. Show all steps."
    enc = tok(prompt, return_tensors='pt')['input_ids'].cuda()

    # ── Correctness check first ──
    print('\n=== correctness sanity (BF16, bs=1, 32 tokens) ===')
    out_hf = model.generate(input_ids=enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
    hf_tokens = out_hf[0, enc.shape[1]:].cpu().tolist()
    print(f'HF generate tokens[:8]:    {hf_tokens[:8]}')

    dec_check = CudaGraphDecoder(model, max_seq_len=enc.shape[1] + 40)
    eager_tokens, _ = dec_check.greedy(enc, max_new=32, use_graph=False)
    print(f'eager  (no graph) tokens[:8]: {eager_tokens[:8]}')
    print(f'eager match HF: {hf_tokens == eager_tokens}')

    graph_tokens, _ = dec_check.greedy(enc, max_new=32, use_graph=True)
    print(f'graph (replay)    tokens[:8]: {graph_tokens[:8]}')
    print(f'graph match eager: {eager_tokens == graph_tokens}')

    print('\n=== BF16 baseline (HF generate) ===')
    for bs in [1, 4, 8]:
        ids = enc.repeat(bs, 1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        print(f'  bs={bs}: {time.perf_counter()-t0:.2f}s')

    print('\n=== BF16 + CudaGraphDecoder (eager mode, no graph) ===')
    dec = CudaGraphDecoder(model, max_seq_len=enc.shape[1] + 70)
    for bs in [1, 4, 8]:
        ids = enc.repeat(bs, 1)
        _ = dec.greedy(ids, max_new=64, use_graph=False)  # warmup
        torch.cuda.synchronize(); t0 = time.perf_counter()
        toks, _ = dec.greedy(ids, max_new=64, use_graph=False)
        torch.cuda.synchronize()
        print(f'  bs={bs}: {time.perf_counter()-t0:.2f}s ({len(toks)} tokens)')

    print('\n=== SRP-FP32 + CudaGraphDecoder (eager mode) ===')
    with method_SRP_FP32(model, ALL_SITES):
        dec2 = CudaGraphDecoder(model, max_seq_len=enc.shape[1] + 70)
        for bs in [1, 4, 8]:
            ids = enc.repeat(bs, 1)
            _ = dec2.greedy(ids, max_new=64, use_graph=False)  # warmup
            torch.cuda.synchronize(); t0 = time.perf_counter()
            toks, _ = dec2.greedy(ids, max_new=64, use_graph=False)
            torch.cuda.synchronize()
            print(f'  bs={bs}: {time.perf_counter()-t0:.2f}s')


if __name__ == '__main__':
    _selftest()
