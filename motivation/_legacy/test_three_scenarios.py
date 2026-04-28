#!/usr/bin/env python3
"""
测试三种场景：
1. Exact tie + 不同的batch size
2. Near tie + 相同的batch size
3. Near tie + 不同的batch size
"""

import sys
import os
import hashlib
from typing import List, Tuple

import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motivation.test_topk_tie_breaking import (
    create_tied_router_scores,
    hash_expert_ids,
)


def run_scenario_1_exact_tie_varying_batch(
    num_experts: int = 8,
    top_k: int = 2,
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    seq_len: int = 100,
    verbose: bool = True,
) -> dict:
    """
    场景1: Exact tie + 不同的batch size
    
    Args:
        num_experts: expert数量
        top_k: top-k值
        num_runs: 运行次数
        min_batch_size: 最小batch size
        max_batch_size: 最大batch size
        seq_len: sequence长度（固定）
        verbose: 是否输出详细信息
    
    Returns:
        包含测试结果的字典
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA GPU")
    
    device = torch.device("cuda")
    
    if verbose:
        print("=" * 80)
        print("SCENARIO 1: Exact Tie + Varying Batch Sizes")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  tie_pattern: k_boundary (exact tie)")
        print(f"  num_experts: {num_experts}")
        print(f"  top_k: {top_k}")
        print(f"  num_runs: {num_runs}")
        print(f"  batch_size range: [{min_batch_size}, {max_batch_size}]")
        print(f"  seq_len: {seq_len} (fixed)")
        print()
    
    # 创建exact tie的base router scores
    base_router_scores = create_tied_router_scores(
        num_tokens=1,
        num_experts=num_experts,
        top_k=top_k,
        tie_pattern="k_boundary",
        tie_value=0.5,
    )[0]  # shape: (num_experts,)
    
    if verbose:
        print("Step 1: Creating exact tie router scores...")
        print(f"  Base router scores: {base_router_scores.cpu().numpy()}")
        sorted_scores, _ = torch.sort(base_router_scores, descending=True)
        print(f"  Sorted scores: {sorted_scores[:top_k+2].cpu().numpy()}")
        if torch.allclose(sorted_scores[top_k-1], sorted_scores[top_k], atol=1e-10):
            print(f"  ✅ Exact tie confirmed: score[{top_k-1}] == score[{top_k}]")
        print()
    
    # 设置随机种子（虽然使用不同的batch size，但保持基础分数相同）
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    if verbose:
        print("Step 2: Running topk with varying batch sizes...")
        print(f"  Running {num_runs} runs...")
    
    results = []
    hash_counts = {}
    batch_size_to_hashes = {}  # 记录每个batch size对应的hash
    
    for run_idx in range(num_runs):
        if verbose and run_idx > 0 and run_idx % 200 == 0:
            print(f"    Progress: {run_idx}/{num_runs} ({run_idx/num_runs*100:.1f}%)")
        
        # 随机选择batch size
        batch_size = random.randint(min_batch_size, max_batch_size)
        
        # 创建router scores: (batch_size, seq_len, num_experts)
        # 每个位置都使用完全相同的base_router_scores（exact tie）
        router_scores = base_router_scores.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, num_experts
        ).clone()
        
        # 同步
        torch.cuda.synchronize()
        
        # 运行topk
        router_scores_flat = router_scores.reshape(-1, num_experts)
        topk_values, topk_indices = torch.topk(router_scores_flat, k=top_k, dim=-1)
        
        torch.cuda.synchronize()
        
        # 计算hash（只对第一个token）
        first_token_indices = topk_indices[0:1]
        hash_value = hash_expert_ids(first_token_indices)
        
        results.append((hash_value, batch_size))
        
        # 统计
        if hash_value not in hash_counts:
            hash_counts[hash_value] = 0
        hash_counts[hash_value] += 1
        
        # 记录batch size与hash的关联
        if batch_size not in batch_size_to_hashes:
            batch_size_to_hashes[batch_size] = set()
        batch_size_to_hashes[batch_size].add(hash_value)
    
    # 分析结果
    unique_hashes = set(hash_counts.keys())
    num_unique_hashes = len(unique_hashes)
    is_non_deterministic = num_unique_hashes > 1
    
    if verbose:
        print()
        print("Step 3: Results Analysis")
        print("=" * 80)
        print(f"  Total runs: {num_runs}")
        print(f"  Unique hash values: {num_unique_hashes}")
        print(f"  Non-deterministic: {'✅ YES' if is_non_deterministic else '❌ NO'}")
        print()
        
        if is_non_deterministic:
            print("  ⚠️  NON-DETERMINISTIC BEHAVIOR DETECTED!")
            print()
            print("  Hash distribution:")
            for hash_val, count in sorted(hash_counts.items(), key=lambda x: -x[1]):
                print(f"    {hash_val[:16]}... : {count} times ({count/num_runs*100:.1f}%)")
            print()
            print("  Batch size vs Hash analysis:")
            for batch_size in sorted(batch_size_to_hashes.keys()):
                hashes = batch_size_to_hashes[batch_size]
                print(f"    batch_size={batch_size}: {len(hashes)} unique hash(es): {[h[:8]+'...' for h in hashes]}")
        else:
            print("  ✅ All runs produced identical expert selections")
            print(f"     (Even with different batch sizes from {min_batch_size} to {max_batch_size})")
        
        print("=" * 80)
        print()
    
    return {
        "scenario": "exact_tie_varying_batch",
        "is_non_deterministic": is_non_deterministic,
        "num_unique_hashes": num_unique_hashes,
        "num_runs": num_runs,
        "hash_counts": hash_counts,
        "batch_size_to_hashes": {k: list(v) for k, v in batch_size_to_hashes.items()},
    }


def run_scenario_2_near_tie_same_batch(
    num_experts: int = 8,
    top_k: int = 2,
    num_runs: int = 1000,
    batch_size: int = 32,
    seq_len: int = 100,
    verbose: bool = True,
) -> dict:
    """
    场景2: Near tie + 相同的batch size
    
    Args:
        num_experts: expert数量
        top_k: top-k值
        num_runs: 运行次数
        batch_size: batch size（固定）
        seq_len: sequence长度（固定）
        verbose: 是否输出详细信息
    
    Returns:
        包含测试结果的字典
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA GPU")
    
    device = torch.device("cuda")
    
    if verbose:
        print("=" * 80)
        print("SCENARIO 2: Near Tie + Same Batch Size")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  tie_pattern: near_tie (close but not exactly equal)")
        print(f"  num_experts: {num_experts}")
        print(f"  top_k: {top_k}")
        print(f"  num_runs: {num_runs}")
        print(f"  batch_size: {batch_size} (fixed)")
        print(f"  seq_len: {seq_len} (fixed)")
        print()
    
    # 创建near tie的base router scores（固定seed，确保base相同）
    base_router_scores = create_tied_router_scores(
        num_tokens=1,
        num_experts=num_experts,
        top_k=top_k,
        tie_pattern="near_tie",
        tie_value=0.5,
    )[0]  # shape: (num_experts,)
    
    if verbose:
        print("Step 1: Creating near tie router scores...")
        print(f"  Base router scores: {base_router_scores.cpu().numpy()}")
        sorted_scores, _ = torch.sort(base_router_scores, descending=True)
        print(f"  Sorted scores: {sorted_scores[:top_k+2].cpu().numpy()}")
        diff = abs(sorted_scores[top_k-1].item() - sorted_scores[top_k].item())
        print(f"  ✅ Near tie confirmed: score[{top_k-1}] vs score[{top_k}], diff={diff:.2e}")
        print(f"     (Values are close but not exactly equal)")
        print()
    
    # 设置随机种子（用于后续的扰动）
    import random
    import numpy as np
    
    base_seed = 42
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    
    if verbose:
        print("Step 2: Running topk with same batch size...")
        print(f"  Running {num_runs} runs...")
        print(f"  Note: Each run uses the SAME base scores, but may add small")
        print(f"        random perturbations to simulate FP precision differences")
        print()
    
    results = []
    hash_counts = {}
    
    # 计算扰动大小（基于base scores的差异）
    sorted_base, _ = torch.sort(base_router_scores, descending=True)
    if top_k > 0 and top_k < num_experts - 1:
        min_diff = (sorted_base[top_k-1] - sorted_base[top_k]).abs().item()
        # 扰动应该足够大，以至于可能改变相对顺序
        # 使用差异的0.5-1.0倍，确保有足够的范围来改变顺序
        epsilon = max(1e-7, min_diff * 0.5)  # 扰动为最小差异的50%
    else:
        epsilon = 1e-6
    
    for run_idx in range(num_runs):
        if verbose and run_idx > 0 and run_idx % 200 == 0:
            print(f"    Progress: {run_idx}/{num_runs} ({run_idx/num_runs*100:.1f}%)")
        
        # 为每次运行添加微小的随机扰动（模拟不同运行中的FP差异）
        # 但保持base相同
        rng = np.random.RandomState(base_seed + run_idx)
        
        # 创建router scores: (batch_size, seq_len, num_experts)
        router_scores = base_router_scores.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, num_experts
        ).clone()
        
        # 为每个位置添加微小的随机扰动（只对前top_k+1个expert）
        perturbation_shape = (batch_size, seq_len, min(top_k + 1, num_experts))
        perturbations = rng.uniform(-epsilon, epsilon, perturbation_shape)
        perturbations_tensor = torch.from_numpy(perturbations).to(
            device=device, dtype=router_scores.dtype
        )
        router_scores[:, :, :perturbation_shape[2]] += perturbations_tensor
        
        # 重新归一化
        router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)
        
        # 同步
        torch.cuda.synchronize()
        
        # 运行topk
        router_scores_flat = router_scores.reshape(-1, num_experts)
        topk_values, topk_indices = torch.topk(router_scores_flat, k=top_k, dim=-1)
        
        torch.cuda.synchronize()
        
        # 计算hash（只对第一个token）
        first_token_indices = topk_indices[0:1]
        hash_value = hash_expert_ids(first_token_indices)
        
        results.append(hash_value)
        
        # 统计
        if hash_value not in hash_counts:
            hash_counts[hash_value] = 0
        hash_counts[hash_value] += 1
    
    # 分析结果
    unique_hashes = set(hash_counts.keys())
    num_unique_hashes = len(unique_hashes)
    is_non_deterministic = num_unique_hashes > 1
    
    if verbose:
        print()
        print("Step 3: Results Analysis")
        print("=" * 80)
        print(f"  Total runs: {num_runs}")
        print(f"  Unique hash values: {num_unique_hashes}")
        print(f"  Non-deterministic: {'✅ YES' if is_non_deterministic else '❌ NO'}")
        print()
        
        if is_non_deterministic:
            print("  ⚠️  NON-DETERMINISTIC BEHAVIOR DETECTED!")
            print()
            print("  Hash distribution:")
            for hash_val, count in sorted(hash_counts.items(), key=lambda x: -x[1]):
                print(f"    {hash_val[:16]}... : {count} times ({count/num_runs*100:.1f}%)")
        else:
            print("  ✅ All runs produced identical expert selections")
            print(f"     (Even with small random perturbations on near-tie values)")
        
        print("=" * 80)
        print()
    
    return {
        "scenario": "near_tie_same_batch",
        "is_non_deterministic": is_non_deterministic,
        "num_unique_hashes": num_unique_hashes,
        "num_runs": num_runs,
        "hash_counts": hash_counts,
    }


def run_scenario_3_near_tie_varying_batch(
    num_experts: int = 8,
    top_k: int = 2,
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    seq_len: int = 100,
    verbose: bool = True,
) -> dict:
    """
    场景3: Near tie + 不同的batch size
    
    Args:
        num_experts: expert数量
        top_k: top-k值
        num_runs: 运行次数
        min_batch_size: 最小batch size
        max_batch_size: 最大batch size
        seq_len: sequence长度（固定）
        verbose: 是否输出详细信息
    
    Returns:
        包含测试结果的字典
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA GPU")
    
    device = torch.device("cuda")
    
    if verbose:
        print("=" * 80)
        print("SCENARIO 3: Near Tie + Varying Batch Sizes")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  tie_pattern: near_tie (close but not exactly equal)")
        print(f"  num_experts: {num_experts}")
        print(f"  top_k: {top_k}")
        print(f"  num_runs: {num_runs}")
        print(f"  batch_size range: [{min_batch_size}, {max_batch_size}]")
        print(f"  seq_len: {seq_len} (fixed)")
        print()
    
    # 创建near tie的base router scores
    base_router_scores = create_tied_router_scores(
        num_tokens=1,
        num_experts=num_experts,
        top_k=top_k,
        tie_pattern="near_tie",
        tie_value=0.5,
    )[0]  # shape: (num_experts,)
    
    if verbose:
        print("Step 1: Creating near tie router scores...")
        print(f"  Base router scores: {base_router_scores.cpu().numpy()}")
        sorted_scores, _ = torch.sort(base_router_scores, descending=True)
        print(f"  Sorted scores: {sorted_scores[:top_k+2].cpu().numpy()}")
        diff = abs(sorted_scores[top_k-1].item() - sorted_scores[top_k].item())
        print(f"  ✅ Near tie confirmed: score[{top_k-1}] vs score[{top_k}], diff={diff:.2e}")
        print(f"     (Values are close but not exactly equal)")
        print()
    
    # 设置随机种子
    import random
    import numpy as np
    
    base_seed = 42
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    
    if verbose:
        print("Step 2: Running topk with varying batch sizes...")
        print(f"  Running {num_runs} runs...")
        print(f"  Note: Each run uses different batch size, and adds small")
        print(f"        random perturbations to simulate FP precision differences")
        print()
    
    results = []
    hash_counts = {}
    batch_size_to_hashes = {}
    
    # 计算扰动大小
    sorted_base, _ = torch.sort(base_router_scores, descending=True)
    if top_k > 0 and top_k < num_experts - 1:
        min_diff = (sorted_base[top_k-1] - sorted_base[top_k]).abs().item()
        # 扰动应该足够大，以至于可能改变相对顺序
        # 使用差异的0.5-1.0倍，确保有足够的范围来改变顺序
        epsilon = max(1e-7, min_diff * 0.5)  # 扰动为最小差异的50%
    else:
        epsilon = 1e-6
    
    for run_idx in range(num_runs):
        if verbose and run_idx > 0 and run_idx % 200 == 0:
            print(f"    Progress: {run_idx}/{num_runs} ({run_idx/num_runs*100:.1f}%)")
        
        # 随机选择batch size
        batch_size = random.randint(min_batch_size, max_batch_size)
        
        # 为每次运行添加微小的随机扰动（使用run_idx和batch_size作为种子的一部分）
        # 这样不同batch size会有不同的扰动模式
        rng = np.random.RandomState(base_seed + run_idx + batch_size * 1000)
        
        # 创建router scores: (batch_size, seq_len, num_experts)
        router_scores = base_router_scores.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, num_experts
        ).clone()
        
        # 为每个位置添加微小的随机扰动（只对前top_k+1个expert）
        # 不同batch size可能有不同的扰动强度（模拟并行度影响）
        perturbation_shape = (batch_size, seq_len, min(top_k + 1, num_experts))
        perturbations = rng.uniform(-epsilon, epsilon, perturbation_shape)
        
        # batch size越大，可能扰动略大（模拟更多并行运算的误差累积）
        perturbations = perturbations * (1 + batch_size * 0.001)
        
        perturbations_tensor = torch.from_numpy(perturbations).to(
            device=device, dtype=router_scores.dtype
        )
        router_scores[:, :, :perturbation_shape[2]] += perturbations_tensor
        
        # 重新归一化
        router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)
        
        # 同步
        torch.cuda.synchronize()
        
        # 运行topk
        router_scores_flat = router_scores.reshape(-1, num_experts)
        topk_values, topk_indices = torch.topk(router_scores_flat, k=top_k, dim=-1)
        
        torch.cuda.synchronize()
        
        # 计算hash（只对第一个token）
        first_token_indices = topk_indices[0:1]
        hash_value = hash_expert_ids(first_token_indices)
        
        results.append((hash_value, batch_size))
        
        # 统计
        if hash_value not in hash_counts:
            hash_counts[hash_value] = 0
        hash_counts[hash_value] += 1
        
        # 记录batch size与hash的关联
        if batch_size not in batch_size_to_hashes:
            batch_size_to_hashes[batch_size] = set()
        batch_size_to_hashes[batch_size].add(hash_value)
    
    # 分析结果
    unique_hashes = set(hash_counts.keys())
    num_unique_hashes = len(unique_hashes)
    is_non_deterministic = num_unique_hashes > 1
    
    if verbose:
        print()
        print("Step 3: Results Analysis")
        print("=" * 80)
        print(f"  Total runs: {num_runs}")
        print(f"  Unique hash values: {num_unique_hashes}")
        print(f"  Non-deterministic: {'✅ YES' if is_non_deterministic else '❌ NO'}")
        print()
        
        if is_non_deterministic:
            print("  ⚠️  NON-DETERMINISTIC BEHAVIOR DETECTED!")
            print()
            print("  Hash distribution:")
            for hash_val, count in sorted(hash_counts.items(), key=lambda x: -x[1]):
                print(f"    {hash_val[:16]}... : {count} times ({count/num_runs*100:.1f}%)")
            print()
            print("  Batch size vs Hash analysis:")
            for batch_size in sorted(batch_size_to_hashes.keys())[:10]:  # 只显示前10个
                hashes = batch_size_to_hashes[batch_size]
                print(f"    batch_size={batch_size}: {len(hashes)} unique hash(es): {[h[:8]+'...' for h in list(hashes)[:3]]}")
            if len(batch_size_to_hashes) > 10:
                print(f"    ... (and {len(batch_size_to_hashes) - 10} more batch sizes)")
        else:
            print("  ✅ All runs produced identical expert selections")
            print(f"     (Even with different batch sizes and random perturbations)")
        
        print("=" * 80)
        print()
    
    return {
        "scenario": "near_tie_varying_batch",
        "is_non_deterministic": is_non_deterministic,
        "num_unique_hashes": num_unique_hashes,
        "num_runs": num_runs,
        "hash_counts": hash_counts,
        "batch_size_to_hashes": {k: list(v) for k, v in batch_size_to_hashes.items()},
    }


def main():
    """运行所有三个场景的测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test three scenarios for topk tie breaking")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 0], default=0,
                       help="Which scenario to run (1, 2, 3, or 0 for all)")
    parser.add_argument("--num-runs", type=int, default=1000, help="Number of runs per scenario")
    parser.add_argument("--min-batch-size", type=int, default=1, help="Minimum batch size")
    parser.add_argument("--max-batch-size", type=int, default=64, help="Maximum batch size")
    parser.add_argument("--batch-size", type=int, default=32, help="Fixed batch size for scenario 2")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k value")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a CUDA GPU.")
        sys.exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print()
    
    results = {}
    
    if args.scenario == 0 or args.scenario == 1:
        print("\n" + "=" * 80)
        print("RUNNING SCENARIO 1")
        print("=" * 80)
        results[1] = run_scenario_1_exact_tie_varying_batch(
            num_experts=args.num_experts,
            top_k=args.top_k,
            num_runs=args.num_runs,
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size,
            seq_len=args.seq_len,
            verbose=not args.quiet,
        )
    
    if args.scenario == 0 or args.scenario == 2:
        print("\n" + "=" * 80)
        print("RUNNING SCENARIO 2")
        print("=" * 80)
        results[2] = run_scenario_2_near_tie_same_batch(
            num_experts=args.num_experts,
            top_k=args.top_k,
            num_runs=args.num_runs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            verbose=not args.quiet,
        )
    
    if args.scenario == 0 or args.scenario == 3:
        print("\n" + "=" * 80)
        print("RUNNING SCENARIO 3")
        print("=" * 80)
        results[3] = run_scenario_3_near_tie_varying_batch(
            num_experts=args.num_experts,
            top_k=args.top_k,
            num_runs=args.num_runs,
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size,
            seq_len=args.seq_len,
            verbose=not args.quiet,
        )
    
    # 总结
    if not args.quiet:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for scenario_num, result in results.items():
            scenario_name = result["scenario"]
            is_non_det = result["is_non_deterministic"]
            num_unique = result["num_unique_hashes"]
            num_runs = result["num_runs"]
            
            status = "✅ NON-DETERMINISTIC" if is_non_det else "❌ DETERMINISTIC"
            print(f"\nScenario {scenario_num} ({scenario_name}):")
            print(f"  Status: {status}")
            print(f"  Unique hashes: {num_unique} out of {num_runs} runs")
        
        print("=" * 80)
    
    # 返回码
    all_non_det = all(r["is_non_deterministic"] for r in results.values())
    if all_non_det:
        sys.exit(0)  # 所有场景都检测到非确定性
    else:
        sys.exit(1)  # 至少有一个场景是确定性的


if __name__ == "__main__":
    main()

