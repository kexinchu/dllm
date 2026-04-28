#!/usr/bin/env python3
"""
直接运行 MoE gating determinism 测试的脚本

根据图片内容，修改为 Motivation Test 2: 真实 Qwen-MoE 推理中, '近似并列'有多普遍

用法:
    # 运行 Motivation Test 2 (测试 near ties 和 non-deterministic 行为)
    python motivation/run_moe_gating_test.py --test motivation2
    
    # 运行原有的 batch invariance 测试
    python motivation/run_moe_gating_test.py --test batch_invariance
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motivation.moe_gating_determinism_test import (
    MoEGatingDeterminismTest,
    _seed_everything,
)

if __name__ == "__main__":
    import torch
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE Gating Determinism Tests")
    parser.add_argument(
        "--test",
        type=str,
        choices=["motivation2", "batch_invariance"],
        default="motivation2",
        help="Which test to run: 'motivation2' (near ties and non-determinism) or 'batch_invariance' (original test)",
    )
    parser.add_argument("--tau", type=float, default=1e-3, help="Near tie threshold (for motivation2)")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for determinism test (for motivation2)")
    parser.add_argument("--save-plots", action="store_true", help="Save distribution plots (for motivation2)")
    parser.add_argument("--plot-dir", type=str, default="motivation/plots", help="Plot directory (for motivation2)")
    parser.add_argument("--sharegpt-json", type=str, default=None, 
                       help="Path to ShareGPT JSON file (for motivation2, default: auto-detect)")
    parser.add_argument("--num-queries", type=int, default=1000, 
                       help="Number of queries to select from ShareGPT (for motivation2, default: 1000)")
    parser.add_argument("--max-query-tokens", type=int, default=2048, 
                       help="Maximum tokens per query (for motivation2, default: 2048)")
    parser.add_argument("--query-seed", type=int, default=42, 
                       help="Random seed for query selection (for motivation2, default: 42)")
    parser.add_argument("--model-path", type=str, default=None, help="Model path")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MoE Gating Determinism Test")
    print("=" * 80)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a CUDA GPU.")
        sys.exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA device count: {torch.cuda.device_count()}")
    print()
    
    # 根据选择的测试运行相应的测试
    if args.test == "motivation2":
        # 运行 Motivation Test 2
        print("=" * 80)
        print("Running Motivation Test 2: Near Ties Prevalence and Non-Deterministic Behavior")
        print("=" * 80)
        print()
        
        try:
            from motivation.test_motivation_2_near_ties import run_motivation_test_2
            
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
            
            # 总结结果
            print("\n" + "=" * 80)
            print("TEST SUMMARY")
            print("=" * 80)
            
            # Non-deterministic 行为
            non_det_result = results["non_deterministic_result"]
            is_non_det = non_det_result["is_non_deterministic"]
            print(f"Non-deterministic behavior: {'✅ YES' if is_non_det else '❌ NO'}")
            if is_non_det:
                print(f"  Inconsistencies found: {len(non_det_result['inconsistencies'])}")
                print(f"  This indicates run-to-run variability in router scores!")
            
            # Near ties 统计
            overall_stats = results["overall_statistics"]
            if overall_stats:
                near_tie_ratio = overall_stats["near_tie_ratio"]
                print(f"\nNear tie prevalence:")
                print(f"  Overall P(Δ < {args.tau}): {near_tie_ratio*100:.2f}%")
                print(f"  Early tokens (<32): {overall_stats['early_near_tie_ratio']*100:.2f}%")
                print(f"  Late tokens (>=32): {overall_stats['late_near_tie_ratio']*100:.2f}%")
                
                # 判据
                if near_tie_ratio >= 1e-3:
                    print(f"\n⚠️  CRITERION MET: P(Δ < {args.tau}) = {near_tie_ratio*100:.2f}% >= 0.1%")
                    print(f"   Top-K flips are likely to be triggered in real systems")
                    print(f"   and will be amplified in the generation layer.")
                else:
                    print(f"\n✅ CRITERION NOT MET: P(Δ < {args.tau}) = {near_tie_ratio*100:.4f}% < 0.1%")
            
            # 按类别显示结果
            category_results = results["category_results"]
            if category_results:
                print(f"\nNear tie ratio by workload category:")
                for category, data in category_results.items():
                    if "statistics" in data:
                        stats = data["statistics"]
                        print(f"  {category}: {stats['near_tie_ratio']*100:.2f}% ({stats['total_tokens']} tokens)")
            
            print("=" * 80)
            
            # 返回码
            if is_non_det or (overall_stats and overall_stats["near_tie_ratio"] >= 1e-3):
                print("\n✅ TEST PASSED: Non-deterministic behavior detected or criterion met")
                sys.exit(0)
            else:
                print("\n⚠️  TEST INCONCLUSIVE: No non-determinism detected and criterion not met")
                sys.exit(1)
        
        except ImportError as e:
            print(f"\n❌ Import error: {e}")
            print("   Make sure test_motivation_2_near_ties.py is available")
            sys.exit(1)
        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ TEST ERROR")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # 运行原有的 batch invariance 测试
        print("=" * 80)
        print("Running test: test_first_moe_layer_batch_invariance_with_random_requests")
        print("=" * 80)
        print()
        
        # 检查 batch_invariant_ops
        try:
            from batch_invariant_ops import set_batch_invariant_mode
            print("✅ batch_invariant_ops available")
        except ImportError:
            print("❌ batch_invariant_ops not available")
            print("   Install via: pip install git+https://github.com/thinking-machines-lab/batch_invariant_ops")
            sys.exit(1)
        
        # 设置测试类
        test_class = MoEGatingDeterminismTest()
        test_class.setUpClass()
        
        try:
            # 运行测试
            test_class.test_first_moe_layer_batch_invariance_with_random_requests()
            print("\n" + "=" * 80)
            print("✅ TEST PASSED")
            print("=" * 80)
            sys.exit(0)
        except AssertionError as e:
            print("\n" + "=" * 80)
            print("❌ TEST FAILED")
            print("=" * 80)
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ TEST ERROR")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

