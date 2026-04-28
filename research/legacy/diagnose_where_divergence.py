"""Capture per-step logits at bs=1 vs bs=8 and find where they first differ.

If even at step 0 (after prefill) logits[0] differ, a primitive op other than
matmul is breaking row-invariance. Helps identify the remaining source.
"""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "."); sys.path.insert(0, "../FP32")

import torch
import torch.nn.functional as F

import determ_llm
import layercast


MODEL = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
N_DECODE = 150


def capture(model, tokenizer, prompt, bs, n_decode):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    ids = ids.repeat(bs, 1).contiguous()
    logits_seq = []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        l = out.logits[0, -1].clone()
        logits_seq.append(l)
        tok = l.argmax().item()
        for _ in range(n_decode):
            next_col = torch.full((bs, 1), tok, dtype=ids.dtype, device=ids.device)
            out = model(input_ids=next_col, past_key_values=past, use_cache=True)
            past = out.past_key_values
            l = out.logits[0, -1].clone()
            logits_seq.append(l)
            tok = l.argmax().item()
    return logits_seq


def compare(label, ls1, ls8):
    print(f"\n[{label}]")
    for step in range(len(ls1)):
        l1 = ls1[step]; l8 = ls8[step]
        diff = (l1 - l8).abs().max().item()
        argmax_match = (l1.argmax().item() == l8.argmax().item())
        if diff > 0 or step < 3:
            print(f"  step={step:>3} max|diff|={diff:.3e} argmax_match={argmax_match}")
            if diff > 0 and step > 0:
                # don't spam — only first few divergences
                if step > 5:
                    break
    if all((ls1[s] - ls8[s]).abs().max().item() == 0 for s in range(len(ls1))):
        print("  ALL STEPS BIT-EXACT")


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
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 7 × 8? Think step by step and put answer in \\boxed{}."}],
        tokenize=False, add_generation_prompt=True,
    )

    # BF16 baseline — no patch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    ls1 = capture(model, tokenizer, prompt, 1, N_DECODE)
    ls8 = capture(model, tokenizer, prompt, 8, N_DECODE)
    compare("BF16 baseline", ls1, ls8)

    # LayerCast (new, full patch)
    layercast.enable()
    ls1 = capture(model, tokenizer, prompt, 1, N_DECODE)
    ls8 = capture(model, tokenizer, prompt, 8, N_DECODE)
    layercast.disable()
    compare("LayerCast (F.linear + matmul + bmm + Tensor.matmul + @)", ls1, ls8)

    # DetermLLM Triton + attn
    determ_llm.enable(backend="triton", attn=True)
    ls1 = capture(model, tokenizer, prompt, 1, N_DECODE)
    ls8 = capture(model, tokenizer, prompt, 8, N_DECODE)
    determ_llm.disable()
    compare("DetermLLM Triton + attn", ls1, ls8)


if __name__ == "__main__":
    main()
