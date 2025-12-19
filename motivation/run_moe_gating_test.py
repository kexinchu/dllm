#!/usr/bin/env python3
"""
直接运行 MoE gating determinism 测试的脚本

用法:
    python motivation/run_moe_gating_test.py
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
    
    print("=" * 60)
    print("MoE Gating Determinism Test")
    print("=" * 60)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a CUDA GPU.")
        sys.exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA device count: {torch.cuda.device_count()}")
    
    # 检查 batch_invariant_ops
    try:
        from batch_invariant_ops import set_batch_invariant_mode
        print("✅ batch_invariant_ops available")
    except ImportError:
        print("❌ batch_invariant_ops not available")
        print("   Install via: pip install git+https://github.com/thinking-machines-lab/batch_invariant_ops")
        sys.exit(1)
    
    # 运行测试
    print("\n" + "=" * 60)
    print("Running test: test_first_moe_layer_batch_invariance_with_random_requests")
    print("=" * 60 + "\n")
    
    # 设置测试类
    test_class = MoEGatingDeterminismTest()
    test_class.setUpClass()
    
    try:
        # 运行测试
        test_class.test_first_moe_layer_batch_invariance_with_random_requests()
        print("\n" + "=" * 60)
        print("✅ TEST PASSED")
        print("=" * 60)
    except AssertionError as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST ERROR")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

