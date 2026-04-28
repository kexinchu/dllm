#!/usr/bin/env python3
"""
Motivation Test 2: 真实 Qwen-MoE 推理中, '近似并列'有多普遍 (决定实际影响上限)

目的: 量化 'k 与 k+1 的 margin 有多小', margin 越小, 越容易被任何微扰 
(batching、cache slicing、不同 kernel) 翻转。

根据图片要求：
1. 在 Qwen-MoE 上跑一组代表性 workload (对话、长文、代码、RAG 查询等)
2. 同时采集每个 token 的 router score (或 softmax 后 gate prob)
3. 对每个 token, 计算:
   - Δ = score[k] - score[k+1] (按降序)
   - tie_count = #{i | score[i] == score[k]} (可选)
   - near_tie = I(Δ < τ), τ 可以设置成与数值误差同量级 (例如 BF16/FP16 的 ulp 量级或经验阈值: 1e-4~1e-3)
4. 画 Δ 的分布 (CDF/直方图), 报告:
   - P(Δ<τ) 在不同 token 位置 (早期 token vs 后期 token) 的比例
   - 在不同 prompt 类别上的差异
5. 判据: 若 P(Δ < τ) 不是极小 (例如达到 1e-3~1e-2 的 token 比例), 那么 Top-K 翻转就很可能在现实系统里触发, 
   并且会在生成层被放大。

同时测试 non-deterministic 行为: 多次运行相同输入，检查 router scores 是否一致。
"""

import sys
import os
import json
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch

# 尝试导入 matplotlib 和 seaborn（可选，用于绘图）
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting will be disabled.")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motivation.moe_gating_determinism_test import (
    _load_model_and_tokenizer,
    _collect_router_captures,
    _seed_everything,
    DEFAULT_MODEL_PATH,
)


# 代表性 workload 类别
WORKLOADS = {
    "dialogue": [
        "Explain what batch-invariant inference means in one sentence.",
        "What are the key advantages of mixture of experts models?",
        "How does attention mechanism work in transformer models?",
        "Describe the difference between training and inference in neural networks.",
        "What is the purpose of softmax normalization in neural networks?",
    ],
    "long_text": [
        "Write a comprehensive explanation of how neural networks learn from data. " * 5,
        "Explain the history and evolution of artificial intelligence from its early beginnings to modern deep learning systems. " * 5,
        "Describe the architecture and components of transformer models in detail. " * 5,
    ],
    "code": [
        "Write a Python function to sort a list of integers in ascending order.",
        "Implement a binary search algorithm in Python.",
        "Create a class in Python to represent a linked list with methods to add and remove nodes.",
        "Write a function to find the longest common subsequence between two strings.",
        "Implement a stack data structure with push, pop, and peek operations.",
    ],
    "rag_query": [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What are the three laws of motion?",
        "Explain the theory of relativity.",
        "What is the difference between DNA and RNA?",
    ],
}


def load_queries_from_sharegpt(
    json_path: str,
    num_queries: int = 1000,
    max_tokens: int = 2048,
    tokenizer=None,
    seed: int = 42,
    verbose: bool = True,
) -> List[str]:
    """
    从 ShareGPT JSON 文件中加载 query。
    
    Args:
        json_path: ShareGPT JSON 文件路径
        num_queries: 要选择的 query 数量（默认 1000）
        max_tokens: 最大 token 数量（默认 2048）
        tokenizer: tokenizer 对象，用于检查长度
        seed: 随机种子
        verbose: 是否输出详细信息
    
    Returns:
        query 列表
    """
    if verbose:
        print(f"Loading queries from {json_path}...")
    
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if verbose:
        print(f"  Total entries in file: {len(data)}")
    
    # 提取所有 human queries
    all_queries = []
    for entry in data:
        if 'conversations' not in entry:
            continue
        
        # 提取第一个 human message 作为 query
        for conv in entry['conversations']:
            if conv.get('from') == 'human' and 'value' in conv:
                query = conv['value'].strip()
                if query:  # 只添加非空 query
                    all_queries.append(query)
                break  # 只取第一个 human message
    
    if verbose:
        print(f"  Extracted {len(all_queries)} human queries")
    
    # 如果有 tokenizer，过滤掉长度超过 max_tokens 的 query
    if tokenizer is not None:
        filtered_queries = []
        for query in all_queries:
            # 检查 token 数量
            tokens = tokenizer(query, return_tensors="pt", truncation=False, add_special_tokens=False)
            num_tokens = tokens['input_ids'].shape[1]
            if num_tokens <= max_tokens:
                filtered_queries.append(query)
        
        if verbose:
            print(f"  Filtered to {len(filtered_queries)} queries with <= {max_tokens} tokens")
        all_queries = filtered_queries
    
    # 随机选择 num_queries 个 query
    random.seed(seed)
    if len(all_queries) < num_queries:
        if verbose:
            print(f"  Warning: Only {len(all_queries)} queries available, using all of them")
        selected_queries = all_queries
    else:
        selected_queries = random.sample(all_queries, num_queries)
        if verbose:
            print(f"  Randomly selected {len(selected_queries)} queries (seed={seed})")
    
    return selected_queries


def compute_margin_and_near_tie(
    probs: torch.Tensor,
    top_k: int,
    tau: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算每个 token 的 margin (Δ) 和 near_tie 指标。
    
    Args:
        probs: shape (batch, seq_len, num_experts) 的 softmax 概率
        top_k: top-k 值
        tau: near_tie 阈值 (默认 1e-3)
    
    Returns:
        delta: shape (batch, seq_len) 的 margin 值: score[k] - score[k+1]
        near_tie: shape (batch, seq_len) 的布尔值: I(Δ < τ)
        tie_count: shape (batch, seq_len) 的整数: #{i | score[i] == score[k]}
    """
    batch_size, seq_len, num_experts = probs.shape
    
    # 对每个位置按降序排序
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # Δ = score[k] - score[k+1] (按降序)
    # k-th 是索引 top_k-1, k+1-th 是索引 top_k
    if top_k < num_experts:
        delta = sorted_probs[:, :, top_k - 1] - sorted_probs[:, :, top_k]
    else:
        # 如果 top_k >= num_experts, 则 delta = 0
        delta = torch.zeros(batch_size, seq_len, device=probs.device)
    
    # near_tie = I(Δ < τ)
    near_tie = (delta < tau).float()
    
    # tie_count = #{i | score[i] == score[k]} (计算与 k-th 值相等的数量)
    kth_value = sorted_probs[:, :, top_k - 1:top_k]  # shape: (batch, seq_len, 1)
    # 计算与 k-th 值相等的数量（允许小的浮点误差）
    tie_count = (torch.abs(probs - kth_value) < 1e-6).sum(dim=-1).float()
    
    return delta, near_tie, tie_count


def analyze_near_ties(
    captures: Dict,
    top_k: int = 2,
    tau: float = 1e-3,
    verbose: bool = True,
) -> Dict:
    """
    分析 captures 中的 near ties 情况。
    
    Args:
        captures: _collect_router_captures 返回的结果
        top_k: top-k 值
        tau: near_tie 阈值
        verbose: 是否输出详细信息
    
    Returns:
        包含分析结果的字典
    """
    results = {
        "deltas": [],  # 所有 token 的 Δ 值
        "near_tie_flags": [],  # 所有 token 的 near_tie 标志
        "tie_counts": [],  # 所有 token 的 tie_count
        "token_positions": [],  # token 位置 (early vs late)
        "layer_indices": [],  # 层索引
        "batch_indices": [],  # batch 索引
    }
    
    # 提取每个层的数据
    for layer_idx, layer_data in captures["layers"].items():
        if "probs" not in layer_data:
            continue
        
        probs = layer_data["probs"]  # shape: (batch, seq_len, num_experts)
        batch_size, seq_len, num_experts = probs.shape
        
        # 计算 margin 和 near_tie
        delta, near_tie, tie_count = compute_margin_and_near_tie(
            probs, top_k=top_k, tau=tau
        )
        
        # 收集所有 token 的数据
        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                results["deltas"].append(delta[batch_idx, token_idx].item())
                results["near_tie_flags"].append(near_tie[batch_idx, token_idx].item())
                results["tie_counts"].append(tie_count[batch_idx, token_idx].item())
                results["token_positions"].append(token_idx)
                results["layer_indices"].append(layer_idx)
                results["batch_indices"].append(batch_idx)
    
    # 转换为 numpy arrays
    results["deltas"] = np.array(results["deltas"])
    results["near_tie_flags"] = np.array(results["near_tie_flags"])
    results["tie_counts"] = np.array(results["tie_counts"])
    results["token_positions"] = np.array(results["token_positions"])
    results["layer_indices"] = np.array(results["layer_indices"])
    results["batch_indices"] = np.array(results["batch_indices"])
    
    # 计算统计信息
    total_tokens = len(results["deltas"])
    near_tie_count = results["near_tie_flags"].sum()
    near_tie_ratio = near_tie_count / total_tokens if total_tokens > 0 else 0.0
    
    results["statistics"] = {
        "total_tokens": total_tokens,
        "near_tie_count": int(near_tie_count),
        "near_tie_ratio": near_tie_ratio,
        "tau": tau,
        "min_delta": float(results["deltas"].min()) if total_tokens > 0 else 0.0,
        "max_delta": float(results["deltas"].max()) if total_tokens > 0 else 0.0,
        "mean_delta": float(results["deltas"].mean()) if total_tokens > 0 else 0.0,
        "median_delta": float(np.median(results["deltas"])) if total_tokens > 0 else 0.0,
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Near Tie Analysis Results (τ={tau})")
        print(f"{'='*80}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Near ties (Δ < {tau}): {near_tie_count} ({near_tie_ratio*100:.2f}%)")
        print(f"  Δ statistics:")
        print(f"    Min: {results['statistics']['min_delta']:.6e}")
        print(f"    Max: {results['statistics']['max_delta']:.6e}")
        print(f"    Mean: {results['statistics']['mean_delta']:.6e}")
        print(f"    Median: {results['statistics']['median_delta']:.6e}")
        print(f"{'='*80}")
    
    return results


def test_near_tie_prompts_with_varying_batch_sizes(
    model,
    tokenizer,
    near_tie_prompts_info: List[Dict],
    top_k: int = 2,
    tau: float = 1e-3,
    batch_sizes: List[int] = [1, 2, 4, 8, 16],
    num_runs_per_batch_size: int = 10,
    max_tokens: int = 2048,
    verbose: bool = True,
) -> Dict:
    """
    对 near tie prompt 进行不同 batch_size + 多次执行的测试。
    
    目的: 检查对于这些 near tie 的 prompt，不同 batch_size + 多次执行
    会不会导致出现选中的 top-k 变化，并记录变化的比例。
    
    Args:
        model: 模型
        tokenizer: tokenizer
        near_tie_prompts_info: near tie prompt 信息列表，每个元素包含 prompt 和 analysis
        top_k: top-k 值
        tau: near_tie 阈值
        batch_sizes: 要测试的 batch_size 列表
        num_runs_per_batch_size: 每个 batch_size 运行的次数
        max_tokens: 最大 token 数量
        verbose: 是否输出详细信息
    
    Returns:
        包含测试结果的字典
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing Near Tie Prompts with Varying Batch Sizes")
        print(f"{'='*80}")
        print(f"  Number of near tie prompts: {len(near_tie_prompts_info)}")
        print(f"  Batch sizes to test: {batch_sizes}")
        print(f"  Runs per batch size: {num_runs_per_batch_size}")
        print(f"  Total test configurations: {len(near_tie_prompts_info)} prompts × {len(batch_sizes)} batch_sizes × {num_runs_per_batch_size} runs")
        print()
    
    if not near_tie_prompts_info:
        if verbose:
            print("  ⚠️  No near tie prompts found, skipping batch size variation test")
        return {
            "total_prompts_tested": 0,
            "prompts_with_topk_changes": 0,
            "overall_topk_change_ratio": 0.0,
            "batch_size_results": {},
            "per_prompt_results": [],
        }
    
    # 限制测试的 prompt 数量（如果太多的话）
    max_prompts_to_test = 50  # 最多测试50个 prompt
    if len(near_tie_prompts_info) > max_prompts_to_test:
        if verbose:
            print(f"  ⚠️  Too many near tie prompts ({len(near_tie_prompts_info)}), limiting to first {max_prompts_to_test}")
        near_tie_prompts_info = near_tie_prompts_info[:max_prompts_to_test]
    
    # 为每个 prompt 存储测试结果
    per_prompt_results = []
    
    # 统计总体结果
    total_token_runs = 0  # 所有 token × batch_size × runs 的总数
    total_topk_changes = 0  # 所有 top-k 变化的次数
    batch_size_results = {bs: {"total_runs": 0, "topk_changes": 0} for bs in batch_sizes}
    
    # 对每个 near tie prompt 进行测试
    for prompt_info_idx, prompt_info in enumerate(near_tie_prompts_info):
        prompt = prompt_info["prompt"]
        prompt_idx = prompt_info["prompt_idx"]
        near_tie_ratio = prompt_info["near_tie_ratio"]
        
        if verbose and (prompt_info_idx + 1) % 10 == 0:
            print(f"  Progress: {prompt_info_idx + 1}/{len(near_tie_prompts_info)} prompts tested ({(prompt_info_idx + 1)/len(near_tie_prompts_info)*100:.1f}%)")
        
        # 为这个 prompt 存储所有运行的结果
        prompt_results = {
            "prompt_idx": prompt_idx,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # 只存储前100个字符
            "near_tie_ratio": near_tie_ratio,
            "batch_size_results": {},
        }
        
        # 对每个 batch_size 进行测试
        for batch_size in batch_sizes:
            # 为这个 batch_size 存储多次运行的结果
            batch_size_topk_results = []  # 存储每次运行的 topk indices
            
            # 多次运行（每次都重置随机种子以确保输入相同）
            for run_idx in range(num_runs_per_batch_size):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    # 重置随机种子（确保输入相同）
                    _seed_everything(42)
                    
                    # 创建 batch（重复相同的 prompt）
                    batch_prompts = [prompt] * batch_size
                    
                    # 收集 router captures
                    capture = _collect_router_captures(
                        model,
                        tokenizer,
                        batch_prompts,
                        max_tokens=max_tokens,
                        capture_inputs=False,
                        use_batch_invariant=False,
                    )
                    
                    # 提取 topk indices（只关注第一个 prompt 的结果）
                    # 从所有层中提取，但通常我们关注第一个 MoE 层
                    for layer_idx, layer_data in capture["layers"].items():
                        if "topk_idx" in layer_data:
                            topk_idx = layer_data["topk_idx"]  # shape: (batch_size, seq_len, top_k)
                            
                            # 只取第一个 batch item（第一个 prompt）
                            first_prompt_topk = topk_idx[0]  # shape: (seq_len, top_k)
                            
                            # 存储这个运行的 topk（转换为 numpy 以便比较）
                            batch_size_topk_results.append({
                                "run": run_idx,
                                "batch_size": batch_size,
                                "layer": layer_idx,
                                "topk_idx": first_prompt_topk.cpu().numpy(),  # 转换为 numpy
                            })
                            
                            total_token_runs += first_prompt_topk.numel()
                            batch_size_results[batch_size]["total_runs"] += first_prompt_topk.numel()
                            
                            break  # 只取第一个层
                
                except Exception as e:
                    if verbose and run_idx < 3:  # 只显示前3个错误
                        print(f"    ⚠️  Error: prompt {prompt_info_idx+1}, batch_size={batch_size}, run={run_idx}: {e}")
                    continue
            
            # 分析这个 batch_size 下多次运行的结果
            # 比较所有运行之间的 topk 是否一致
            if len(batch_size_topk_results) > 0:
                baseline_run = batch_size_topk_results[0]
                baseline_topk = baseline_run["topk_idx"]
                
                num_changes_in_batch_size = 0
                num_comparisons_in_batch_size = 0
                
                for other_run in batch_size_topk_results[1:]:
                    other_topk = other_run["topk_idx"]
                    
                    if baseline_topk.shape == other_topk.shape:
                        # 比较 topk indices
                        changes = (baseline_topk != other_topk).sum()
                        num_changes_in_batch_size += changes
                        num_comparisons_in_batch_size += baseline_topk.size
                
                prompt_results["batch_size_results"][batch_size] = {
                    "num_runs": len(batch_size_topk_results),
                    "num_changes": num_changes_in_batch_size,
                    "num_comparisons": num_comparisons_in_batch_size,
                    "change_ratio": float(num_changes_in_batch_size / num_comparisons_in_batch_size) if num_comparisons_in_batch_size > 0 else 0.0,
                }
                
                total_topk_changes += num_changes_in_batch_size
                batch_size_results[batch_size]["topk_changes"] += num_changes_in_batch_size
        
        # 计算这个 prompt 在所有 batch_size 下的总体变化比例
        prompt_total_changes = 0
        prompt_total_comparisons = 0
        for bs in batch_sizes:
            if bs in prompt_results["batch_size_results"]:
                bs_result = prompt_results["batch_size_results"][bs]
                prompt_total_changes += bs_result["num_changes"]
                prompt_total_comparisons += bs_result["num_comparisons"]
        
        prompt_results["overall_change_ratio"] = (
            float(prompt_total_changes / prompt_total_comparisons) if prompt_total_comparisons > 0 else 0.0
        )
        prompt_results["has_topk_changes"] = prompt_total_changes > 0
        
        per_prompt_results.append(prompt_results)
    
    # 计算总体统计
    overall_topk_change_ratio = (
        float(total_topk_changes / total_token_runs) if total_token_runs > 0 else 0.0
    )
    
    prompts_with_topk_changes = sum(1 for r in per_prompt_results if r["has_topk_changes"])
    
    # 按 batch_size 计算变化比例
    for bs in batch_sizes:
        if batch_size_results[bs]["total_runs"] > 0:
            batch_size_results[bs]["change_ratio"] = (
                float(batch_size_results[bs]["topk_changes"] / batch_size_results[bs]["total_runs"])
            )
        else:
            batch_size_results[bs]["change_ratio"] = 0.0
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Batch Size Variation Test Results")
        print(f"{'='*80}")
        print(f"  Total prompts tested: {len(near_tie_prompts_info)}")
        print(f"  Prompts with top-k changes: {prompts_with_topk_changes} ({prompts_with_topk_changes/len(near_tie_prompts_info)*100:.1f}%)")
        print(f"  Overall top-k change ratio: {overall_topk_change_ratio*100:.2f}%")
        print(f"    (Total changes: {total_topk_changes} / Total comparisons: {total_token_runs})")
        print()
        print(f"  Change ratio by batch size:")
        for bs in batch_sizes:
            bs_result = batch_size_results[bs]
            if bs_result["total_runs"] > 0:
                print(f"    batch_size={bs}: {bs_result['change_ratio']*100:.2f}% "
                      f"({bs_result['topk_changes']}/{bs_result['total_runs']})")
        
        # 显示一些有变化的 prompt 示例
        prompts_with_changes = [r for r in per_prompt_results if r["has_topk_changes"]]
        if prompts_with_changes:
            print(f"\n  Sample prompts with top-k changes (first 5):")
            for i, result in enumerate(prompts_with_changes[:5]):
                print(f"    Prompt {result['prompt_idx']+1}:")
                print(f"      Near tie ratio: {result['near_tie_ratio']*100:.2f}%")
                print(f"      Overall change ratio: {result['overall_change_ratio']*100:.2f}%")
                print(f"      Change ratio by batch size:")
                for bs in batch_sizes:
                    if bs in result["batch_size_results"]:
                        bs_result = result["batch_size_results"][bs]
                        print(f"        batch_size={bs}: {bs_result['change_ratio']*100:.2f}% "
                              f"({bs_result['num_changes']}/{bs_result['num_comparisons']})")
            if len(prompts_with_changes) > 5:
                print(f"    ... and {len(prompts_with_changes) - 5} more prompts with changes")
        
        print(f"{'='*80}")
    
    return {
        "total_prompts_tested": len(near_tie_prompts_info),
        "prompts_with_topk_changes": prompts_with_topk_changes,
        "prompts_with_topk_changes_ratio": float(prompts_with_topk_changes / len(near_tie_prompts_info)) if near_tie_prompts_info else 0.0,
        "overall_topk_change_ratio": overall_topk_change_ratio,
        "total_topk_changes": total_topk_changes,
        "total_token_runs": total_token_runs,
        "batch_size_results": batch_size_results,
        "per_prompt_results": per_prompt_results,
    }


def test_non_deterministic_behavior(
    model,
    tokenizer,
    prompts: List[str],
    num_runs: int = 10,
    tau: float = 1e-3,
    verbose: bool = True,
) -> Dict:
    """
    测试 non-deterministic 行为: 多次运行相同输入，检查 router scores 是否一致。
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompts: 输入的 prompts
        num_runs: 运行次数
        tau: near_tie 阈值
        verbose: 是否输出详细信息
    
    Returns:
        包含 non-deterministic 分析结果的字典
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing Non-Deterministic Behavior")
        print(f"{'='*80}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Num runs: {num_runs}")
        print(f"  Near tie threshold (τ): {tau}")
    
    # 多次运行，收集 captures
    all_captures = []
    
    for run_idx in range(num_runs):
        if verbose and run_idx > 0 and run_idx % 5 == 0:
            print(f"    Progress: {run_idx}/{num_runs} runs completed")
        
        # 每次运行前重置随机种子（确保输入相同）
        _seed_everything(42)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        try:
            capture = _collect_router_captures(
                model,
                tokenizer,
                prompts,
                max_tokens=128,  # 限制 token 数量以节省内存
                capture_inputs=False,
                use_batch_invariant=False,  # 使用标准操作以检测非确定性
            )
            all_captures.append(capture)
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Run {run_idx} failed: {e}")
            continue
    
    if len(all_captures) < 2:
        if verbose:
            print(f"    ❌ Not enough successful runs for comparison")
        return {
            "is_non_deterministic": False,
            "num_runs": len(all_captures),
            "inconsistencies": [],
        }
    
    # 比较多次运行的结果
    baseline = all_captures[0]
    inconsistencies = []
    
    for run_idx in range(1, len(all_captures)):
        compare_capture = all_captures[run_idx]
        
        # 比较每个层的 probs
        for layer_idx in set(baseline["layers"].keys()) & set(compare_capture["layers"].keys()):
            baseline_probs = baseline["layers"][layer_idx].get("probs")
            compare_probs = compare_capture["layers"][layer_idx].get("probs")
            
            if baseline_probs is None or compare_probs is None:
                continue
            
            # 比较形状
            if baseline_probs.shape != compare_probs.shape:
                inconsistencies.append({
                    "run": run_idx,
                    "layer": layer_idx,
                    "issue": "shape_mismatch",
                    "baseline_shape": list(baseline_probs.shape),
                    "compare_shape": list(compare_probs.shape),
                })
                continue
            
            # 比较值（允许小的浮点误差）
            max_diff = (baseline_probs - compare_probs).abs().max().item()
            mean_diff = (baseline_probs - compare_probs).abs().mean().item()
            
            if max_diff > 1e-5:  # 检测明显的差异
                # 检查 topk indices 是否不同
                baseline_topk = baseline["layers"][layer_idx].get("topk_idx")
                compare_topk = compare_capture["layers"][layer_idx].get("topk_idx")
                
                topk_different = False
                num_topk_different = 0
                total_topk = 0
                
                if baseline_topk is not None and compare_topk is not None:
                    if baseline_topk.shape == compare_topk.shape:
                        topk_different = not torch.equal(baseline_topk, compare_topk)
                        if topk_different:
                            num_topk_different = (baseline_topk != compare_topk).sum().item()
                            total_topk = baseline_topk.numel()
                
                # 计算 near tie 比例（可能影响 non-determinism）
                delta_baseline, near_tie_baseline, _ = compute_margin_and_near_tie(
                    baseline_probs, top_k=2, tau=tau
                )
                near_tie_ratio_baseline = near_tie_baseline.float().mean().item()
                
                inconsistencies.append({
                    "run": run_idx,
                    "layer": layer_idx,
                    "issue": "value_mismatch",
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "topk_different": topk_different,
                    "num_topk_different": num_topk_different,
                    "total_topk": total_topk,
                    "near_tie_ratio": near_tie_ratio_baseline,
                })
    
    is_non_deterministic = len(inconsistencies) > 0
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Non-Deterministic Behavior Analysis")
        print(f"{'='*80}")
        print(f"  Total runs compared: {len(all_captures)}")
        print(f"  Inconsistencies found: {len(inconsistencies)}")
        print(f"  Non-deterministic: {'✅ YES' if is_non_deterministic else '❌ NO'}")
        
        if inconsistencies:
            print(f"\n  Inconsistency Details:")
            for inc in inconsistencies[:10]:  # 只显示前10个
                print(f"    Run {inc['run']}, Layer {inc['layer']}:")
                print(f"      Issue: {inc['issue']}")
                if inc['issue'] == 'value_mismatch':
                    print(f"      Max diff: {inc['max_diff']:.6e}")
                    print(f"      Mean diff: {inc['mean_diff']:.6e}")
                    print(f"      Top-k different: {inc['topk_different']}")
                    if inc['num_topk_different'] > 0:
                        pct = (inc['num_topk_different'] / inc['total_topk']) * 100
                        print(f"      Top-k changed: {inc['num_topk_different']}/{inc['total_topk']} ({pct:.1f}%)")
                    print(f"      Near tie ratio: {inc['near_tie_ratio']:.4f}")
            if len(inconsistencies) > 10:
                print(f"    ... and {len(inconsistencies) - 10} more inconsistencies")
        print(f"{'='*80}")
    
    return {
        "is_non_deterministic": is_non_deterministic,
        "num_runs": len(all_captures),
        "inconsistencies": inconsistencies,
    }


def run_motivation_test_2(
    model_path: Optional[str] = None,
    tau: float = 1e-3,
    num_runs_for_determinism: int = 10,
    verbose: bool = True,
    save_plots: bool = True,
    plot_dir: str = "motivation/plots",
    sharegpt_json_path: Optional[str] = None,
    num_queries: int = 1000,
    max_query_tokens: int = 2048,
    query_seed: int = 42,
) -> Dict:
    """
    运行 Motivation Test 2: 分析真实 Qwen-MoE 推理中的 near ties 和 non-deterministic 行为。
    
    Args:
        model_path: 模型路径（默认使用 DEFAULT_MODEL_PATH）
        tau: near_tie 阈值（默认 1e-3）
        num_runs_for_determinism: 测试 non-determinism 的运行次数
        verbose: 是否输出详细信息
        save_plots: 是否保存分布图
        plot_dir: 图表保存目录
        sharegpt_json_path: ShareGPT JSON 文件路径（如果为 None，则使用默认路径或预设的 WORKLOADS）
        num_queries: 从 ShareGPT 中选择的 query 数量（默认 1000）
        max_query_tokens: 最大 query token 数量（默认 2048）
        query_seed: 随机选择 query 的种子（默认 42）
    
    Returns:
        包含所有分析结果的字典
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA GPU")
    
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    
    if verbose:
        print("=" * 80)
        print("Motivation Test 2: Near Ties Prevalence in Real Qwen-MoE Inference")
        print("=" * 80)
        print(f"Model path: {model_path}")
        print(f"Near tie threshold (τ): {tau}")
        print(f"Num runs for determinism test: {num_runs_for_determinism}")
        print()
    
    # 加载模型
    if verbose:
        print("Loading model and tokenizer...")
    tokenizer, model = _load_model_and_tokenizer(model_path)
    model.eval()
    
    if verbose:
        print(f"✅ Model loaded: {torch.cuda.get_device_name(0)}")
        print()
    
    # 获取 top_k 值（从第一个 MoE 层）
    top_k = None
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'top_k'):
            top_k = layer.mlp.top_k
            break
    
    if top_k is None:
        top_k = 2  # 默认值
    
    if verbose:
        print(f"Top-k value: {top_k}")
        print()
    
    # 1. 加载 query（从 ShareGPT 或使用预设的 WORKLOADS）
    if sharegpt_json_path is None:
        # 尝试查找默认路径
        default_paths = [
            "/workspace/dllm/ShareGPT_V3_unfiltered_cleaned_split.json",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "ShareGPT_V3_unfiltered_cleaned_split.json"),
        ]
        sharegpt_json_path = None
        for path in default_paths:
            if os.path.exists(path):
                sharegpt_json_path = path
                break
    
    if sharegpt_json_path and os.path.exists(sharegpt_json_path):
        if verbose:
            print(f"Loading queries from ShareGPT file: {sharegpt_json_path}")
        prompts = load_queries_from_sharegpt(
            sharegpt_json_path,
            num_queries=num_queries,
            max_tokens=max_query_tokens,
            tokenizer=tokenizer,
            seed=query_seed,
            verbose=verbose,
        )
        
        # 将所有 prompts 作为一个类别处理
        category_results = {}
        category_results["sharegpt"] = {
            "deltas": [],
            "near_tie_flags": [],
            "token_positions": [],
        }
    else:
        if verbose:
            print(f"⚠️  ShareGPT file not found, using default WORKLOADS")
            print(f"    Searched paths: {default_paths if sharegpt_json_path is None else [sharegpt_json_path]}")
        # 使用预设的 WORKLOADS（向后兼容）
        prompts = []
        for category, cat_prompts in WORKLOADS.items():
            prompts.extend(cat_prompts)
        category_results = {}
        for category in WORKLOADS.keys():
            category_results[category] = {
                "deltas": [],
                "near_tie_flags": [],
                "token_positions": [],
            }
    
    if verbose:
        print(f"\nTotal prompts to process: {len(prompts)}")
        print()
    
    # 2. 运行所有 prompts 并收集 router scores
    if verbose:
        print(f"{'='*80}")
        print(f"Processing {len(prompts)} prompts...")
        print(f"{'='*80}")
    
    # 存储每个 prompt 的详细信息，用于后续识别 near tie prompt
    prompt_results = []  # 存储 (prompt, capture, analysis) 的列表
    
    # 对每个 prompt 运行
    for prompt_idx, prompt in enumerate(prompts):
        if verbose and (prompt_idx + 1) % 100 == 0:
            print(f"  Progress: {prompt_idx + 1}/{len(prompts)} prompts processed ({(prompt_idx + 1)/len(prompts)*100:.1f}%)")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # 重置随机种子确保可重复性
            _seed_everything(42)
            
            # 收集 router captures
            capture = _collect_router_captures(
                model,
                tokenizer,
                [prompt],
                max_tokens=max_query_tokens,  # 使用 max_query_tokens 作为限制
                capture_inputs=False,
                use_batch_invariant=False,
            )
            
            # 分析 near ties
            analysis = analyze_near_ties(
                capture,
                top_k=top_k,
                tau=tau,
                verbose=False,  # 不在每个 prompt 输出详细信息
            )
            
            # 存储 prompt 的结果（用于后续分析）
            prompt_results.append({
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "capture": capture,
                "analysis": analysis,
            })
            
            # 合并结果
            if sharegpt_json_path and os.path.exists(sharegpt_json_path):
                # 使用单一类别
                category_key = "sharegpt"
            else:
                # 根据 prompt 来源确定类别（简单启发式）
                category_key = "dialogue"  # 默认类别
                for cat, cat_prompts in WORKLOADS.items():
                    if prompt in cat_prompts:
                        category_key = cat
                        break
            
            if category_key not in category_results:
                category_results[category_key] = {
                    "deltas": [],
                    "near_tie_flags": [],
                    "token_positions": [],
                }
            
            category_results[category_key]["deltas"].extend(analysis["deltas"].tolist())
            category_results[category_key]["near_tie_flags"].extend(analysis["near_tie_flags"].tolist())
            category_results[category_key]["token_positions"].extend(analysis["token_positions"].tolist())
            
        except Exception as e:
            if verbose and prompt_idx < 10:  # 只显示前10个错误
                print(f"    ⚠️  Error processing prompt {prompt_idx + 1}: {e}")
            continue
    
    # 3. 计算每个类别的统计信息
    for category_key in category_results.keys():
        if category_results[category_key]["deltas"]:
            deltas = np.array(category_results[category_key]["deltas"])
            near_tie_flags = np.array(category_results[category_key]["near_tie_flags"])
            token_positions = np.array(category_results[category_key]["token_positions"])
            
            near_tie_ratio = near_tie_flags.mean()
            
            # 按 token 位置分组（early vs late）
            early_mask = token_positions < 32
            late_mask = token_positions >= 32
            
            early_near_tie_ratio = near_tie_flags[early_mask].mean() if early_mask.sum() > 0 else 0.0
            late_near_tie_ratio = near_tie_flags[late_mask].mean() if late_mask.sum() > 0 else 0.0
            
            category_results[category_key]["statistics"] = {
                "total_tokens": len(deltas),
                "near_tie_ratio": float(near_tie_ratio),
                "early_near_tie_ratio": float(early_near_tie_ratio),
                "late_near_tie_ratio": float(late_near_tie_ratio),
                "min_delta": float(deltas.min()),
                "max_delta": float(deltas.max()),
                "mean_delta": float(deltas.mean()),
                "median_delta": float(np.median(deltas)),
            }
            
            if verbose:
                print(f"\n  Category '{category_key}' statistics:")
                print(f"    Total tokens: {len(deltas)}")
                print(f"    Near tie ratio (P(Δ < {tau})): {near_tie_ratio*100:.2f}%")
                print(f"    Early tokens (<32): {early_near_tie_ratio*100:.2f}%")
                print(f"    Late tokens (>=32): {late_near_tie_ratio*100:.2f}%")
    
    # 4. 识别 near tie prompt，并对它们进行不同 batch_size 的测试
    near_tie_prompts_info = []  # 存储有 near tie 的 prompt 信息
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Identifying Near Tie Prompts")
        print(f"{'='*80}")
    
    # 找出有 near tie token 的 prompt
    for prompt_result in prompt_results:
        prompt = prompt_result["prompt"]
        analysis = prompt_result["analysis"]
        
        # 检查是否有任何 token 是 near tie
        near_tie_flags = analysis["near_tie_flags"]
        has_near_tie = np.any(near_tie_flags > 0.5)  # 检查是否有任何 token 是 near tie
        
        if has_near_tie:
            near_tie_ratio = near_tie_flags.mean()
            near_tie_prompts_info.append({
                "prompt": prompt,
                "prompt_idx": prompt_result["prompt_idx"],
                "near_tie_ratio": near_tie_ratio,
                "analysis": analysis,
            })
    
    if verbose:
        print(f"  Found {len(near_tie_prompts_info)} prompts with near ties (out of {len(prompt_results)} total)")
        if near_tie_prompts_info:
            avg_near_tie_ratio = np.mean([info["near_tie_ratio"] for info in near_tie_prompts_info])
            print(f"  Average near tie ratio in these prompts: {avg_near_tie_ratio*100:.2f}%")
        print()
    
    # 4.1 对 near tie prompt 进行不同 batch_size + 多次执行的测试
    batch_size_variation_result = None
    if near_tie_prompts_info:
        batch_size_variation_result = test_near_tie_prompts_with_varying_batch_sizes(
            model,
            tokenizer,
            near_tie_prompts_info,
            top_k=top_k,
            tau=tau,
            batch_sizes=[1, 2, 4, 8, 16],  # 测试不同的 batch_size
            num_runs_per_batch_size=10,  # 每个 batch_size 运行 10 次
            max_tokens=max_query_tokens,
            verbose=verbose,
        )
    
    # 4.2 测试 non-deterministic 行为（使用第一个 prompt）
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing Non-Deterministic Behavior")
        print(f"{'='*80}")
    
    test_prompt = prompts[0] if prompts else "Test prompt"
    non_det_result = test_non_deterministic_behavior(
        model,
        tokenizer,
        [test_prompt],
        num_runs=num_runs_for_determinism,
        tau=tau,
        verbose=verbose,
    )
    
    # 5. 汇总所有结果
    overall_stats = {}
    if category_results:
        all_deltas = []
        all_near_tie_flags = []
        all_token_positions = []
        
        for category, data in category_results.items():
            if "deltas" in data:
                all_deltas.extend(data["deltas"])
                all_near_tie_flags.extend(data["near_tie_flags"])
                all_token_positions.extend(data["token_positions"])
        
        if all_deltas:
            all_deltas = np.array(all_deltas)
            all_near_tie_flags = np.array(all_near_tie_flags)
            all_token_positions = np.array(all_token_positions)
            
            overall_near_tie_ratio = all_near_tie_flags.mean()
            
            # 按 token 位置分组
            early_mask = all_token_positions < 32
            late_mask = all_token_positions >= 32
            
            early_near_tie_ratio = all_near_tie_flags[early_mask].mean() if early_mask.sum() > 0 else 0.0
            late_near_tie_ratio = all_near_tie_flags[late_mask].mean() if late_mask.sum() > 0 else 0.0
            
            overall_stats = {
                "total_tokens": len(all_deltas),
                "near_tie_ratio": float(overall_near_tie_ratio),
                "early_near_tie_ratio": float(early_near_tie_ratio),
                "late_near_tie_ratio": float(late_near_tie_ratio),
                "min_delta": float(all_deltas.min()),
                "max_delta": float(all_deltas.max()),
                "mean_delta": float(all_deltas.mean()),
                "median_delta": float(np.median(all_deltas)),
            }
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"OVERALL STATISTICS")
                print(f"{'='*80}")
                print(f"  Total tokens analyzed: {overall_stats['total_tokens']}")
                print(f"  Overall near tie ratio (P(Δ < {tau})): {overall_stats['near_tie_ratio']*100:.2f}%")
                print(f"  Early tokens (<32): {overall_stats['early_near_tie_ratio']*100:.2f}%")
                print(f"  Late tokens (>=32): {overall_stats['late_near_tie_ratio']*100:.2f}%")
                print(f"  Δ statistics:")
                print(f"    Min: {overall_stats['min_delta']:.6e}")
                print(f"    Max: {overall_stats['max_delta']:.6e}")
                print(f"    Mean: {overall_stats['mean_delta']:.6e}")
                print(f"    Median: {overall_stats['median_delta']:.6e}")
                
                # 判断是否符合判据
                if overall_stats['near_tie_ratio'] >= 1e-3:
                    print(f"\n  ⚠️  CRITERION MET: P(Δ < {tau}) = {overall_stats['near_tie_ratio']*100:.2f}% >= 0.1%")
                    print(f"     Top-K flips are likely to be triggered in real systems")
                    print(f"     and will be amplified in the generation layer.")
                else:
                    print(f"\n  ✅ CRITERION NOT MET: P(Δ < {tau}) = {overall_stats['near_tie_ratio']*100:.4f}% < 0.1%")
                print(f"{'='*80}")
    
    # 6. 保存分布图（如果启用且 matplotlib 可用）
    if save_plots and category_results and HAS_PLOTTING:
        os.makedirs(plot_dir, exist_ok=True)
        
        # 合并所有 deltas 用于绘图
        all_deltas_plot = []
        for category, data in category_results.items():
            if "deltas" in data:
                all_deltas_plot.extend(data["deltas"])
        
        if all_deltas_plot:
            all_deltas_plot = np.array(all_deltas_plot)
            
            # 过滤掉负值（理论上不应该有，但为了安全）
            all_deltas_plot = all_deltas_plot[all_deltas_plot >= 0]
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 直方图
            axes[0, 0].hist(all_deltas_plot, bins=100, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(tau, color='r', linestyle='--', label=f'τ = {tau}')
            axes[0, 0].set_xlabel('Δ = score[k] - score[k+1]')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Δ (Margin)')
            axes[0, 0].set_xscale('log')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. CDF
            sorted_deltas = np.sort(all_deltas_plot)
            y = np.arange(1, len(sorted_deltas) + 1) / len(sorted_deltas)
            axes[0, 1].plot(sorted_deltas, y, linewidth=2)
            axes[0, 1].axvline(tau, color='r', linestyle='--', label=f'τ = {tau}')
            axes[0, 1].set_xlabel('Δ = score[k] - score[k+1]')
            axes[0, 1].set_ylabel('CDF')
            axes[0, 1].set_title('CDF of Δ (Margin)')
            axes[0, 1].set_xscale('log')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 按类别比较
            categories_list = list(category_results.keys())
            near_tie_ratios = [
                category_results[cat].get("statistics", {}).get("near_tie_ratio", 0.0) * 100
                for cat in categories_list
            ]
            axes[1, 0].bar(categories_list, near_tie_ratios, alpha=0.7, edgecolor='black')
            axes[1, 0].axhline(0.1, color='r', linestyle='--', label='Threshold (0.1%)')
            axes[1, 0].set_ylabel('Near Tie Ratio (%)')
            axes[1, 0].set_title('Near Tie Ratio by Workload Category')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # 4. Early vs Late tokens
            if overall_stats:
                early_late_data = [
                    overall_stats['early_near_tie_ratio'] * 100,
                    overall_stats['late_near_tie_ratio'] * 100,
                ]
                axes[1, 1].bar(['Early (<32)', 'Late (>=32)'], early_late_data, alpha=0.7, edgecolor='black')
                axes[1, 1].axhline(0.1, color='r', linestyle='--', label='Threshold (0.1%)')
                axes[1, 1].set_ylabel('Near Tie Ratio (%)')
                axes[1, 1].set_title('Near Tie Ratio by Token Position')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, "motivation_test_2_near_ties.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"\n✅ Plot saved to: {plot_path}")
    elif save_plots and not HAS_PLOTTING:
        if verbose:
            print(f"\n⚠️  Plotting disabled: matplotlib/seaborn not available")
    
    # 确保 top_k 被包含在结果中
    return {
        "overall_statistics": overall_stats,
        "category_results": category_results,
        "non_deterministic_result": non_det_result,
        "batch_size_variation_result": batch_size_variation_result,
        "near_tie_prompts_count": len(near_tie_prompts_info) if near_tie_prompts_info else 0,
        "tau": tau,
        "top_k": top_k if 'top_k' in locals() else 2,
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Motivation Test 2: Near Ties Prevalence")
    parser.add_argument("--model-path", type=str, default=None, help="Model path")
    parser.add_argument("--tau", type=float, default=1e-3, help="Near tie threshold")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for determinism test")
    parser.add_argument("--save-plots", action="store_true", help="Save distribution plots")
    parser.add_argument("--plot-dir", type=str, default="motivation/plots", help="Plot directory")
    parser.add_argument("--sharegpt-json", type=str, default=None, 
                       help="Path to ShareGPT JSON file (default: auto-detect)")
    parser.add_argument("--num-queries", type=int, default=1000, 
                       help="Number of queries to select from ShareGPT (default: 1000)")
    parser.add_argument("--max-query-tokens", type=int, default=2048, 
                       help="Maximum tokens per query (default: 2048)")
    parser.add_argument("--query-seed", type=int, default=42, 
                       help="Random seed for query selection (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    try:
        results = run_motivation_test_2(
            model_path=args.model_path,
            tau=args.tau,
            num_runs_for_determinism=args.num_runs,
            verbose=not args.quiet,
            save_plots=args.save_plots,
            plot_dir=args.plot_dir,
            sharegpt_json_path=args.sharegpt_json,
            num_queries=args.num_queries,
            max_query_tokens=args.max_query_tokens,
            query_seed=args.query_seed,
        )
        
        # 返回码: 如果检测到 non-deterministic 或 near_tie_ratio >= 1e-3，返回 0
        is_non_det = results["non_deterministic_result"]["is_non_deterministic"]
        near_tie_ratio = results.get("overall_statistics", {}).get("near_tie_ratio", 0.0)
        
        if is_non_det or near_tie_ratio >= 1e-3:
            print("\n✅ TEST PASSED: Non-deterministic behavior detected or criterion met")
            sys.exit(0)
        else:
            print("\n⚠️  TEST INCONCLUSIVE: No non-determinism detected and criterion not met")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

