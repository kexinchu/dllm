"""Smoke test: confirm that LayerCast and DetermLLM-hybrid with attention
patches produce batch-invariant outputs on a real decoder.

Runs DeepSeek-R1-Distill-Qwen-7B on one short MATH500 prompt at bs=1 and
bs=8 (same prompt repeated), then compares token hashes.
"""
import hashlib
import os
import sys
import time

import torch
import torch.nn.functional as F

DLLM = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(DLLM, "..", "FP32")
sys.path.insert(0, DLLM)
sys.path.insert(0, FP32)

import determ_llm
import layercast

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
MODEL = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
GEN_LEN = 300  # long enough to trigger the S≥128 batch-dependent bucket
PROMPT = (
    "Let $f(x) = x^2 + 3x + 2$. What is $f(5)$? "
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def generate(model, tokenizer, prompt_text, bs, gen_len, device):
    torch.manual_seed(0)
    enc = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    ids = enc.repeat(bs, 1).contiguous()
    toks = []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        tok = F.log_softmax(logits, dim=-1).argmax().item()
        toks.append(tok)
        for _ in range(gen_len - 1):
            next_col = torch.full((bs, 1), tok, dtype=ids.dtype, device=device)
            out = model(input_ids=next_col, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            tok = F.log_softmax(logits, dim=-1).argmax().item()
            toks.append(tok)
            if tok == tokenizer.eos_token_id:
                break
    return toks


def run_case(label, enable_fn, disable_fn, model, tokenizer, device):
    enable_fn()
    t0 = time.time()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    toks_1 = generate(model, tokenizer, prompt, 1, GEN_LEN, device)
    toks_8 = generate(model, tokenizer, prompt, 8, GEN_LEN, device)
    disable_fn()
    h1 = hashlib.md5(str(toks_1).encode()).hexdigest()[:10]
    h8 = hashlib.md5(str(toks_8).encode()).hexdigest()[:10]
    # first divergence
    div = next((j for j, (a, b) in enumerate(zip(toks_1, toks_8)) if a != b),
               min(len(toks_1), len(toks_8)))
    dt = time.time() - t0
    match = "MATCH" if toks_1 == toks_8 else f"DIV@{div}"
    print(f"  [{label}] bs1={h1}  bs8={h8}  {match}  ({dt:.0f}s)")
    return toks_1 == toks_8, div


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading DeepSeek-7B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Loaded on {device}\n")

    print(f"--- gen_len={GEN_LEN}, prompt='{PROMPT[:40]}...' ---")

    # BF16 baseline (no patches)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    run_case("BF16 baseline", lambda: None, lambda: None, model, tokenizer, device)

    # DetermLLM hybrid, attn=False (previous behaviour)
    run_case("DetermLLM hybrid (attn=False, old)",
             lambda: determ_llm.enable(backend="hybrid", attn=False),
             lambda: determ_llm.disable(),
             model, tokenizer, device)

    # DetermLLM hybrid, attn=True (new: FP32-upcast attention)
    run_case("DetermLLM hybrid (attn=True, new)",
             lambda: determ_llm.enable(backend="hybrid", attn=True),
             lambda: determ_llm.disable(),
             model, tokenizer, device)

    # DetermLLM Triton, attn=True  — recommended path
    run_case("DetermLLM Triton (attn=True, new)",
             lambda: determ_llm.enable(backend="triton", attn=True),
             lambda: determ_llm.disable(),
             model, tokenizer, device)

    # LayerCast (new: patches F.linear + torch.matmul + torch.bmm)
    run_case("LayerCast (F.linear + matmul + bmm, new)",
             lambda: layercast.enable(),
             lambda: layercast.disable(),
             model, tokenizer, device)


if __name__ == "__main__":
    main()
