"""M2: RMSNorm and Softmax chunk reduction variance, wide sweep."""
import torch, json
torch.manual_seed(42)

device = 'cuda'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_m2_reduction.json'
results = {'rmsnorm': [], 'softmax': []}

# ===== RMSNorm =====
# Simulate different chunk reduction orders for sum-of-squares
print("=== RMSNorm Chunk Reduction Variance ===")
for dim in [1024, 2048, 4096, 8192]:
    for mode, use_fp32 in [('BF16', False), ('FP32', True)]:
        x = torch.randn(32, dim, device=device, dtype=torch.bfloat16)

        # Reference: single-pass reduction
        if use_fp32:
            ref_var = x.float().pow(2).mean(-1, keepdim=True)
        else:
            ref_var = x.pow(2).mean(-1, keepdim=True).float()

        for n_chunks in [1, 2, 4, 8, 16, 32, 64]:
            if n_chunks > dim:
                continue
            chunk_size = dim // n_chunks
            # Chunked reduction: sum partial variances then combine
            if use_fp32:
                partial = [x[:, i*chunk_size:(i+1)*chunk_size].float().pow(2).sum(-1, keepdim=True)
                          for i in range(n_chunks)]
            else:
                partial = [x[:, i*chunk_size:(i+1)*chunk_size].pow(2).sum(-1, keepdim=True).float()
                          for i in range(n_chunks)]

            # Combine in different orders
            # Order 1: sequential
            total1 = partial[0].clone()
            for p in partial[1:]:
                total1 = total1 + p
            var1 = total1 / dim

            # Order 2: reverse
            total2 = partial[-1].clone()
            for p in reversed(partial[:-1]):
                total2 = total2 + p
            var2 = total2 / dim

            # Order 3: tree reduction
            level = list(partial)
            while len(level) > 1:
                next_level = []
                for i in range(0, len(level), 2):
                    if i + 1 < len(level):
                        next_level.append(level[i] + level[i+1])
                    else:
                        next_level.append(level[i])
                level = next_level
            var3 = level[0] / dim

            # Compare all orders to reference
            d12 = (var1 - var2).abs().max().item()
            d13 = (var1 - var3).abs().max().item()
            d1r = (var1 - ref_var).abs().max().item()

            results['rmsnorm'].append({
                'dim': dim, 'chunks': n_chunks, 'mode': mode,
                'seq_vs_rev': d12, 'seq_vs_tree': d13, 'seq_vs_ref': d1r
            })

        print(f"  dim={dim} {mode}: done")

# ===== Softmax =====
print("\n=== Softmax Chunk Reduction Variance ===")
for vocab in [32000, 128256]:
    for mode, use_fp32 in [('BF16', False), ('FP32', True)]:
        logits = torch.randn(16, vocab, device=device, dtype=torch.bfloat16)

        # Reference softmax
        if use_fp32:
            ref_sm = torch.softmax(logits.float(), dim=-1)
        else:
            ref_sm = torch.softmax(logits, dim=-1).float()

        for n_chunks in [1, 2, 4, 8, 16, 32]:
            if n_chunks > vocab:
                continue
            cs = vocab // n_chunks

            if use_fp32:
                logits_f = logits.float()
            else:
                logits_f = logits

            # Chunked online softmax: compute per-chunk max, then global max
            chunk_maxes = [logits_f[:, i*cs:(i+1)*cs].max(dim=-1, keepdim=True).values for i in range(n_chunks)]
            global_max = torch.cat(chunk_maxes, dim=-1).max(dim=-1, keepdim=True).values

            # Sum of exp with global max
            total_exp = torch.zeros(logits.shape[0], 1, device=device, dtype=torch.float32)
            for i in range(n_chunks):
                chunk = logits_f[:, i*cs:(i+1)*cs]
                total_exp += (chunk.float() - global_max.float()).exp().sum(dim=-1, keepdim=True)

            # Full softmax with same global max
            sm_chunked = ((logits_f.float() - global_max.float()).exp() / total_exp)

            diff = (ref_sm.float() - sm_chunked.float()).abs()
            max_d = diff.max().item()
            kl = (ref_sm.float() * (ref_sm.float().clamp(min=1e-10).log() - sm_chunked.float().clamp(min=1e-10).log())).sum(-1).mean().item()

            results['softmax'].append({
                'vocab': vocab, 'chunks': n_chunks, 'mode': mode,
                'max_diff': max_d, 'kl_div': kl
            })

        print(f"  vocab={vocab} {mode}: done")

# Print summary
print("\n=== RMSNorm Summary (max diff between reduction orders) ===")
print(f"{'dim':>6} {'mode':<5} {'chunks':>7} {'seq_vs_rev':>12} {'seq_vs_tree':>12}")
for r in results['rmsnorm']:
    if r['chunks'] in [1, 4, 16, 64]:
        print(f"{r['dim']:>6} {r['mode']:<5} {r['chunks']:>7} {r['seq_vs_rev']:>12.2e} {r['seq_vs_tree']:>12.2e}")

print("\n=== Softmax Summary ===")
print(f"{'vocab':>8} {'mode':<5} {'chunks':>7} {'max_diff':>12} {'kl_div':>12}")
for r in results['softmax']:
    print(f"{r['vocab']:>8} {r['mode']:<5} {r['chunks']:>7} {r['max_diff']:>12.2e} {r['kl_div']:>12.2e}")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
