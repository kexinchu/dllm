#!/usr/bin/env python3
"""
LayerCast Latency Test: 测试 BF16 存储 + FP32 计算 vs 纯 BF16 的 end-to-end latency

参考论文: Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference
- LayerCast: Weights/bias 使用 BF16 存储，但计算使用 FP32
- 开销: doubling the memory usage and inference time compared to BF16

测试两种场景:
1. 纯 BF16: 存储和计算都使用 BF16
2. LayerCast: 存储使用 BF16，计算使用 FP32

测量 end-to-end latency (包括 tokenization + inference)
"""

import sys
import os
import json
import random
import time
from typing import List, Dict, Optional
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_queries_from_sharegpt(
    json_path: str,
    num_queries: int = 10,
    max_tokens: int = 2048,
    tokenizer=None,
    seed: int = 42,
    verbose: bool = True,
) -> List[str]:
    """
    从 ShareGPT JSON 文件中加载 query。
    
    Args:
        json_path: ShareGPT JSON 文件路径
        num_queries: 要选择的 query 数量（默认 10）
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


def apply_layercast(model: nn.Module):
    """
    应用 LayerCast: 模型权重保持 BF16 存储，但在 forward 时转换为 FP32 进行计算。
    
    通过包装 Linear 层的 forward 方法来实现：
    - 在 forward 时将输入和权重都转换为 FP32
    - 计算完成后将输出转换回 BF16
    """
    original_forwards = {}
    
    def layercast_forward(self, input):
        """LayerCast forward: 权重和输入都转换为 FP32 进行计算"""
        # 保存原始输入类型
        input_is_bf16 = isinstance(input, torch.Tensor) and input.dtype == torch.bfloat16
        
        # 转换输入为 FP32
        if input_is_bf16:
            input_fp32 = input.to(torch.float32)
        else:
            input_fp32 = input
        
        # 转换权重和偏置为 FP32
        weight_fp32 = self.weight.to(torch.float32) if self.weight.dtype == torch.bfloat16 else self.weight
        bias_fp32 = self.bias.to(torch.float32) if self.bias is not None and self.bias.dtype == torch.bfloat16 else self.bias
        
        # 使用 FP32 进行计算
        output_fp32 = torch.nn.functional.linear(input_fp32, weight_fp32, bias_fp32)
        
        # 转换输出回 BF16（如果原始输入是 BF16）
        if input_is_bf16:
            return output_fp32.to(torch.bfloat16)
        return output_fp32
    
    # 为所有 Linear 层应用 LayerCast
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_forwards[name] = module.forward
            # 绑定新的 forward 方法
            import types
            module.forward = types.MethodType(layercast_forward, module)
    
    return original_forwards


def remove_layercast(model: nn.Module, original_forwards: Dict):
    """移除 LayerCast，恢复原始的 forward 方法"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in original_forwards:
            module.forward = original_forwards[name]


@contextmanager
def layercast_mode(model: nn.Module):
    """上下文管理器：临时启用 LayerCast"""
    original_forwards = apply_layercast(model)
    try:
        yield
    finally:
        remove_layercast(model, original_forwards)


def run_inference(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    queries: List[str],
    use_layercast: bool = False,
    max_new_tokens: int = 128,
    num_warmup: int = 2,
    num_runs: int = 3,
    verbose: bool = True,
) -> Dict:
    """
    运行推理并测量 latency。
    
    Args:
        model: 模型
        tokenizer: tokenizer
        queries: query 列表
        use_layercast: 是否使用 LayerCast
        max_new_tokens: 最大生成 token 数
        num_warmup: warmup 次数
        num_runs: 实际测量次数
        verbose: 是否输出详细信息
    
    Returns:
        包含 latency 统计信息的字典
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 应用或移除 LayerCast
    if use_layercast:
        original_forwards = apply_layercast(model)
        mode_name = "LayerCast (BF16 storage + FP32 compute)"
    else:
        original_forwards = {}
        mode_name = "Pure BF16 (BF16 storage + BF16 compute)"
    
    try:
        latencies = []
        tokenization_times = []
        inference_times = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running inference with: {mode_name}")
            print(f"{'='*80}")
            print(f"Number of queries: {len(queries)}")
            print(f"Warmup runs: {num_warmup}, Measurement runs: {num_runs}")
            print()
        
        # Warmup
        if verbose:
            print("Warming up...")
        for _ in range(num_warmup):
            for query in queries[:min(2, len(queries))]:  # 只用前2个query做warmup
                inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )
        
        # 同步 CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if verbose:
            print("Running measurements...")
        
        # 实际测量
        for run_idx in range(num_runs):
            run_latencies = []
            run_tokenization_times = []
            run_inference_times = []
            
            for query_idx, query in enumerate(queries):
                # Tokenization time
                t0 = time.perf_counter()
                inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
                # 将输入移动到模型所在的设备
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                else:
                    # 如果使用 device_map="auto"，找到第一个参数所在的设备
                    model_device = next(model.parameters()).device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                t1 = time.perf_counter()
                tokenization_time = (t1 - t0) * 1000  # ms
                
                # Inference time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                
                inference_time = (t3 - t2) * 1000  # ms
                total_time = (t3 - t0) * 1000  # ms
                
                run_latencies.append(total_time)
                run_tokenization_times.append(tokenization_time)
                run_inference_times.append(inference_time)
                
                if verbose and run_idx == 0:
                    print(f"  Query {query_idx+1}/{len(queries)}: "
                          f"total={total_time:.2f}ms "
                          f"(tokenization={tokenization_time:.2f}ms, "
                          f"inference={inference_time:.2f}ms)")
            
            latencies.append(run_latencies)
            tokenization_times.append(run_tokenization_times)
            inference_times.append(run_inference_times)
        
        # 计算统计信息
        # 每个 query 的平均 latency（跨多次运行）
        avg_latencies = np.mean(latencies, axis=0)
        avg_tokenization_times = np.mean(tokenization_times, axis=0)
        avg_inference_times = np.mean(inference_times, axis=0)
        
        # 总体统计
        all_latencies = np.array(latencies).flatten()
        all_tokenization_times = np.array(tokenization_times).flatten()
        all_inference_times = np.array(inference_times).flatten()
        
        results = {
            "mode": mode_name,
            "num_queries": len(queries),
            "num_runs": num_runs,
            "per_query_avg_latency_ms": avg_latencies.tolist(),
            "per_query_avg_tokenization_ms": avg_tokenization_times.tolist(),
            "per_query_avg_inference_ms": avg_inference_times.tolist(),
            "overall_mean_latency_ms": float(np.mean(all_latencies)),
            "overall_std_latency_ms": float(np.std(all_latencies)),
            "overall_mean_tokenization_ms": float(np.mean(all_tokenization_times)),
            "overall_mean_inference_ms": float(np.mean(all_inference_times)),
            "overall_std_inference_ms": float(np.std(all_inference_times)),
            "total_time_ms": float(np.sum(all_latencies)),
        }
        
        if verbose:
            print(f"\nResults for {mode_name}:")
            print(f"  Overall mean latency: {results['overall_mean_latency_ms']:.2f} ± {results['overall_std_latency_ms']:.2f} ms")
            print(f"  Overall mean tokenization: {results['overall_mean_tokenization_ms']:.2f} ms")
            print(f"  Overall mean inference: {results['overall_mean_inference_ms']:.2f} ± {results['overall_std_inference_ms']:.2f} ms")
            print(f"  Total time: {results['total_time_ms']:.2f} ms")
        
        return results
    
    finally:
        # 移除 LayerCast
        if original_forwards:
            remove_layercast(model, original_forwards)


def main():
    """主函数"""
    import argparse
    import warnings
    warnings.filterwarnings("ignore", message="Token indices sequence length")
    warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
    warnings.filterwarnings("ignore", message="generation flags are not valid")

    parser = argparse.ArgumentParser(description="Test LayerCast latency vs Pure BF16")
    parser.add_argument("--model-path", type=str, 
                       default="/workspace/Models/Llama-3.1-8B-Instruct",
                       help="Path to the model")
    parser.add_argument("--sharegpt-json", type=str,
                       default="/workspace/dllm/ShareGPT_V3_unfiltered_cleaned_split.json",
                       help="Path to ShareGPT JSON file")
    parser.add_argument("--num-queries", type=int, default=10,
                       help="Number of queries to test")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--num-warmup", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of measurement runs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for query selection")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path for results")
    
    args = parser.parse_args()
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a CUDA GPU.")
        sys.exit(1)
    
    print("="*80)
    print("LayerCast Latency Test")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"ShareGPT JSON: {args.sharegpt_json}")
    print(f"Number of queries: {args.num_queries}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # 加载模型和 tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    # 设置 padding token（Llama 模型通常使用 eos_token 作为 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,  # 使用 dtype 而不是 torch_dtype
        device_map="auto",
    )
    model.eval()
    print(f"✅ Model loaded")
    print()
    
    # 加载 queries
    queries = load_queries_from_sharegpt(
        args.sharegpt_json,
        num_queries=args.num_queries,
        tokenizer=tokenizer,
        seed=args.seed,
        verbose=True,
    )
    print()
    
    # 测试纯 BF16
    print("\n" + "="*80)
    print("TEST 1: Pure BF16 (BF16 storage + BF16 compute)")
    print("="*80)
    results_bf16 = run_inference(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        use_layercast=False,
        max_new_tokens=args.max_new_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        verbose=True,
    )
    
    # 清理 CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 测试 LayerCast
    print("\n" + "="*80)
    print("TEST 2: LayerCast (BF16 storage + FP32 compute)")
    print("="*80)
    results_layercast = run_inference(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        use_layercast=True,
        max_new_tokens=args.max_new_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        verbose=True,
    )
    
    # 对比结果
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    bf16_latency = results_bf16['overall_mean_inference_ms']
    layercast_latency = results_layercast['overall_mean_inference_ms']
    speedup = bf16_latency / layercast_latency
    slowdown = layercast_latency / bf16_latency
    
    print(f"Pure BF16 inference latency:     {bf16_latency:.2f} ± {results_bf16['overall_std_inference_ms']:.2f} ms")
    print(f"LayerCast inference latency:     {layercast_latency:.2f} ± {results_layercast['overall_std_inference_ms']:.2f} ms")
    print(f"Slowdown factor:                 {slowdown:.2f}x")
    print(f"Speedup factor:                  {speedup:.2f}x")
    print()
    
    print(f"Pure BF16 total time:            {results_bf16['total_time_ms']:.2f} ms")
    print(f"LayerCast total time:            {results_layercast['total_time_ms']:.2f} ms")
    print()
    
    # 保存结果
    all_results = {
        "config": {
            "model_path": args.model_path,
            "num_queries": args.num_queries,
            "max_new_tokens": args.max_new_tokens,
            "num_runs": args.num_runs,
            "seed": args.seed,
        },
        "pure_bf16": results_bf16,
        "layercast": results_layercast,
        "comparison": {
            "bf16_latency_ms": bf16_latency,
            "layercast_latency_ms": layercast_latency,
            "slowdown_factor": slowdown,
            "speedup_factor": speedup,
        },
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print("Results (use --output to save to file):")
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()

