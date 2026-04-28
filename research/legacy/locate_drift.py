"""OPTION A — systematically locate the first module whose row-0 output
differs across bs={1, 8} under our fully-patched DetermLLM stack.

Procedure:
  1. Enable DetermLLM (Triton F.linear + Triton attention + flag=False).
  2. Run prefill on the same prompt at bs=1 and bs=8, capture row-0
     output of every submodule via forward hooks.
  3. Advance decode step by step up to ``STEPS``, capturing again.
  4. Report the earliest (step, module) where row-0 diverges.

Assumes our kernels are bit-exact per row (Proof C). Any remaining drift
must come from a non-matmul op (softmax, norm reduction, gather, cat, ...).
"""
import os
import sys
from collections import OrderedDict

os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "."); sys.path.insert(0, "../FP32")

import torch
import torch.nn.functional as F

import determ_llm

MODEL = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
PROBLEM_IDX = 0  # which math500 problem to trace
STEPS = 40       # number of decode steps to trace (cover step 15 where Proof D first diverged)
BS_PROBE = 4     # compare bs=1 vs bs=BS_PROBE (bs=4 is the odd one)


def _load_problem(idx):
    import json, os
    path = os.path.join(os.path.dirname(__file__), "math500_cached.json")
    with open(path) as f:
        return json.load(f)[idx]["problem"]

# Per-(step, module_name) capture. Inner dict keyed by bs ∈ {1, BS_PROBE}.
captures = OrderedDict()

_current_step = {'val': 0}
_current_bs = {'val': 1}


def _to_cpu_clone(t):
    if torch.is_tensor(t):
        return t.detach().cpu().clone()
    return None


def _hook_out(module, inp, out, name):
    """Capture row-0 (first batch element) of the output for this module."""
    if isinstance(out, tuple):
        out = out[0]
    if not torch.is_tensor(out):
        return
    key = (_current_step['val'], name)
    if key not in captures:
        captures[key] = {}
    # Only row 0 (identical-input row across bs).
    if out.dim() >= 1:
        captures[key][_current_bs['val']] = _to_cpu_clone(out[0])


def _register_hooks(model):
    """Hook every module including container (DecoderLayer) so we capture
    the final output after any post-module residual adds."""
    hooks = []
    for name, module in model.named_modules():
        if name == "":
            continue
        h = module.register_forward_hook(
            lambda m, i, o, n=name: _hook_out(m, i, o, n))
        hooks.append(h)
    return hooks


def _remove_hooks(hooks):
    for h in hooks:
        h.remove()


def run_prefill_decode(model, tokenizer, prompt_text, bs, steps):
    """Run prefill + ``steps`` decode steps at this batch size.
    Each forward call is tagged with a step index via ``_current_step``."""
    _current_bs['val'] = bs
    enc = tokenizer(prompt_text, return_tensors="pt")["input_ids"].cuda()
    ids = enc.repeat(bs, 1).contiguous()

    with torch.no_grad():
        _current_step['val'] = -1  # prefill
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        tok = logits.argmax().item()

        for step in range(steps):
            _current_step['val'] = step
            nx = torch.full((bs, 1), tok, dtype=ids.dtype, device=ids.device)
            out = model(input_ids=nx, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            tok = logits.argmax().item()
    return


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

    problem = _load_problem(PROBLEM_IDX)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\nPlease reason step by step, "
                                       f"and put your final answer within \\boxed{{}}."}],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"Using math500 problem #{PROBLEM_IDX}: {problem[:80]}...")

    # Configure DetermLLM: Triton F.linear + Triton attention.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    determ_llm.enable(backend="triton", attn=True)

    hooks = _register_hooks(model)
    print(f"Registered {len(hooks)} forward hooks\n")

    try:
        print(f"Running bs=1 ({STEPS+1} forward calls: 1 prefill + {STEPS} decode)...")
        run_prefill_decode(model, tokenizer, prompt_text, 1, STEPS)
        print(f"Running bs={BS_PROBE}...")
        run_prefill_decode(model, tokenizer, prompt_text, BS_PROBE, STEPS)
    finally:
        _remove_hooks(hooks)
        determ_llm.disable()

    # Compare row-0 per-module output.
    print(f"\nComparing {len(captures)} (step, module) captures for bs=1 vs bs={BS_PROBE}:")
    first_drift_per_step = {}
    step_counts = {}
    for (step, name), bs_dict in captures.items():
        step_counts[step] = step_counts.get(step, 0) + 1
        if 1 not in bs_dict or BS_PROBE not in bs_dict:
            continue
        a = bs_dict[1]; b = bs_dict[BS_PROBE]
        if a.shape != b.shape:
            continue
        if not torch.equal(a, b):
            maxd = float((a - b).abs().max())
            if step not in first_drift_per_step:
                first_drift_per_step[step] = (name, maxd)

    print(f"\nTotal modules per step: {step_counts.get(-1, 0)} (prefill), "
          f"{step_counts.get(0, 0)} (decode step 0)")

    print("\n=== First-diverging module per step ===")
    for step in sorted(first_drift_per_step.keys()):
        name, maxd = first_drift_per_step[step]
        label = "prefill" if step == -1 else f"decode step {step}"
        print(f"  {label:18s}  first drift at  {name:60s}  max|diff|={maxd:.3e}")

    # Focused drift map at first-divergence step around the divergent module.
    if first_drift_per_step:
        first_step = min(first_drift_per_step.keys())
        first_name = first_drift_per_step[first_step][0]
        # Find layer index in the divergent module path
        import re
        m = re.search(r"layers\.(\d+)", first_name)
        target_layers = None
        if m:
            k = int(m.group(1))
            target_layers = set(range(max(0, k-2), k+2))

        print(f"\n=== Drift map at step {first_step} around {first_name} ===")
        for (step, name), bs_dict in captures.items():
            if step != first_step:
                continue
            if target_layers is not None:
                m2 = re.search(r"layers\.(\d+)", name)
                if m2 is None or int(m2.group(1)) not in target_layers:
                    continue
            if 1 not in bs_dict or BS_PROBE not in bs_dict:
                continue
            a = bs_dict[1]; b = bs_dict[BS_PROBE]
            if a.shape != b.shape:
                continue
            eq = torch.equal(a, b)
            maxd = float((a - b).abs().max()) if not eq else 0.0
            mark = "OK  " if eq else "DIFF"
            print(f"  {mark}  max|diff|={maxd:.3e}  {name}")


if __name__ == "__main__":
    main()
