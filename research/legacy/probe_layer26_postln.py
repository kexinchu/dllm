"""Hook input AND output of layer 26 post_attention_layernorm at decode
step 10 to determine whether the LayerNorm's INPUT is bit-exact across
bs=1 vs bs=4 (which would mean RMSNorm itself is bs-dependent) or
whether the input is already different (meaning the elementwise residual
add somehow drifts)."""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "."); sys.path.insert(0, "../FP32")

import torch
import torch.nn.functional as F
import determ_llm

MODEL = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
TARGET_STEP = 10


def _load_problem(idx=0):
    import json
    with open(os.path.join(os.path.dirname(__file__), "math500_cached.json")) as f:
        return json.load(f)[idx]["problem"]


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

    problem = _load_problem(0)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\nPlease reason step by step, "
                                      f"and put your final answer within \\boxed{{}}."}],
        tokenize=False, add_generation_prompt=True,
    )

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    determ_llm.enable(backend="triton", attn=True)
    n_patched = determ_llm.patch_rmsnorm(model)
    print(f"Patched {n_patched} RMSNorm modules to bs-invariant Triton kernel")

    target = model.model.layers[26].post_attention_layernorm
    captures = {}  # (bs, kind) -> tensor (row 0)

    state = {'step': -1, 'bs': 1}

    def hook(module, inp, out):
        if state['step'] != TARGET_STEP:
            return
        x = inp[0] if isinstance(inp, tuple) else inp
        captures[(state['bs'], 'in')] = x[0].detach().cpu().clone()
        captures[(state['bs'], 'out')] = out[0].detach().cpu().clone()

    h = target.register_forward_hook(hook)

    def run(bs):
        state['bs'] = bs
        enc = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        ids = enc.repeat(bs, 1).contiguous()
        with torch.no_grad():
            state['step'] = -1
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            tk = out.logits[0, -1].argmax().item()
            for s in range(TARGET_STEP + 1):
                state['step'] = s
                nx = torch.full((bs, 1), tk, dtype=ids.dtype, device=ids.device)
                out = model(input_ids=nx, past_key_values=past, use_cache=True)
                past = out.past_key_values
                tk = out.logits[0, -1].argmax().item()

    run(1)
    run(4)
    h.remove()
    determ_llm.disable()

    in1 = captures[(1, 'in')]; in4 = captures[(4, 'in')]
    out1 = captures[(1, 'out')]; out4 = captures[(4, 'out')]

    print(f"\n=== Layer 26 post_attention_layernorm at decode step {TARGET_STEP} ===")
    print(f"INPUT  bs=1 vs bs=4: bit-exact? {torch.equal(in1, in4)}  "
          f"max|diff|={float((in1.float() - in4.float()).abs().max()):.3e}")
    print(f"OUTPUT bs=1 vs bs=4: bit-exact? {torch.equal(out1, out4)}  "
          f"max|diff|={float((out1.float() - out4.float()).abs().max()):.3e}")
    print(f"\nINPUT shape={in1.shape} dtype={in1.dtype}")
    print(f"OUTPUT shape={out1.shape} dtype={out1.dtype}")

    # If input differs, find indices
    if not torch.equal(in1, in4):
        diff = (in1.float() - in4.float()).abs()
        idx = diff.flatten().argmax().item()
        print(f"\nLargest INPUT diff at flat idx={idx}: bs1={in1.flatten()[idx].item()} bs4={in4.flatten()[idx].item()}")


if __name__ == "__main__":
    main()
