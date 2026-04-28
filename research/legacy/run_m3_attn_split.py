"""M3: Attention split-KV variance — comprehensive sweep."""
import torch, json, math
torch.manual_seed(42)

device = 'cuda'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_m3_attn_split.json'

results = []

def manual_attention_split(Q, K, V, n_splits):
    """Online softmax attention with explicit split-KV."""
    bs, n_heads, seq_q, head_dim = Q.shape
    seq_kv = K.shape[2]
    split_size = math.ceil(seq_kv / n_splits)
    scale = 1.0 / math.sqrt(head_dim)

    # Initialize running stats
    m = torch.full((bs, n_heads, seq_q, 1), float('-inf'), device=device, dtype=torch.float32)
    l = torch.zeros((bs, n_heads, seq_q, 1), device=device, dtype=torch.float32)
    o = torch.zeros((bs, n_heads, seq_q, head_dim), device=device, dtype=torch.float32)

    for i in range(n_splits):
        start = i * split_size
        end = min(start + split_size, seq_kv)
        if start >= seq_kv:
            break

        Ki = K[:, :, start:end, :]
        Vi = V[:, :, start:end, :]

        # Local attention scores
        s = (Q.float() @ Ki.float().transpose(-2, -1)) * scale  # [bs, h, sq, chunk]
        m_local = s.max(dim=-1, keepdim=True).values
        p = (s - m_local).exp()
        l_local = p.sum(dim=-1, keepdim=True)
        o_local = p @ Vi.float()

        # Online softmax update
        m_new = torch.maximum(m, m_local)
        alpha = (m - m_new).exp()
        beta = (m_local - m_new).exp()
        l_new = alpha * l + beta * l_local
        o = (alpha * l * o + beta * o_local) / l_new.clamp(min=1e-10)
        m = m_new
        l = l_new

    return o

# Sweep parameters
head_dim = 128
n_heads = 8
seq_q = 1  # decode phase

print(f"{'seq_kv':>7} {'splits':>7} {'mode':<5} {'vs_1split':>12} {'vs_ref':>12}")
print('-' * 55)

for seq_kv in [64, 128, 256, 512, 1024, 2048, 4096]:
    Q = torch.randn(1, n_heads, seq_q, head_dim, device=device, dtype=torch.bfloat16)
    K = torch.randn(1, n_heads, seq_kv, head_dim, device=device, dtype=torch.bfloat16)
    V = torch.randn(1, n_heads, seq_kv, head_dim, device=device, dtype=torch.bfloat16)

    for mode in ['BF16', 'FP32']:
        # Reference: PyTorch SDPA (no split)
        with torch.no_grad():
            if mode == 'FP32':
                ref = torch.nn.functional.scaled_dot_product_attention(
                    Q.float(), K.float(), V.float()).to(torch.bfloat16)
            else:
                ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

        # 1-split reference (our manual implementation baseline)
        out_1 = manual_attention_split(Q, K, V, 1)

        for n_splits in [1, 2, 4, 8, 16, 32, 64]:
            if n_splits > seq_kv:
                continue
            out_s = manual_attention_split(Q, K, V, n_splits)

            vs_1 = (out_1 - out_s).abs().max().item()
            vs_ref = (ref.float() - out_s).abs().max().item()

            results.append({
                'seq_kv': seq_kv, 'n_splits': n_splits, 'mode': mode,
                'vs_1split': vs_1, 'vs_sdpa_ref': vs_ref,
                'head_dim': head_dim, 'n_heads': n_heads
            })

            print(f"{seq_kv:>7} {n_splits:>7} {mode:<5} {vs_1:>12.4e} {vs_ref:>12.4e}")

    print()

# Cross-split comparison: how much does changing split count matter?
print("\n=== Key comparison: fixed split vs varying split ===")
print(f"{'seq_kv':>7} {'mode':<5} {'1vs2':>10} {'1vs4':>10} {'1vs16':>10} {'1vs32':>10}")
for seq_kv in [256, 1024, 4096]:
    for mode in ['BF16', 'FP32']:
        row = [r for r in results if r['seq_kv']==seq_kv and r['mode']==mode]
        base = [r for r in row if r['n_splits']==1][0]['vs_1split']
        vals = {}
        for ns in [2, 4, 16, 32]:
            match = [r for r in row if r['n_splits']==ns]
            if match:
                vals[ns] = match[0]['vs_1split']
        print(f"{seq_kv:>7} {mode:<5}", end='')
        for ns in [2, 4, 16, 32]:
            if ns in vals:
                print(f" {vals[ns]:>10.4e}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        print()

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
