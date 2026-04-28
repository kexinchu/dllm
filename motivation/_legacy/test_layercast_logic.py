#!/usr/bin/env python3
"""
测试 LayerCast 实现的逻辑正确性（不需要 GPU）
"""

import torch
import torch.nn as nn
from test_layercast_latency import apply_layercast, remove_layercast


def test_layercast_basic():
    """测试基本的 LayerCast 功能"""
    print("Testing basic LayerCast functionality...")
    
    # 创建一个简单的模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    
    # 转换为 BF16
    model = model.to(torch.bfloat16)
    
    # 创建测试输入
    input_tensor = torch.randn(2, 10, dtype=torch.bfloat16)
    
    # 原始输出
    with torch.no_grad():
        original_output = model(input_tensor)
    
    print(f"Original output dtype: {original_output.dtype}")
    print(f"Original output shape: {original_output.shape}")
    print(f"Original output (first 5 values): {original_output[0, :5]}")
    
    # 应用 LayerCast
    original_forwards = apply_layercast(model)
    print(f"\nApplied LayerCast to {len(original_forwards)} Linear layers")
    
    # LayerCast 输出
    with torch.no_grad():
        layercast_output = model(input_tensor)
    
    print(f"LayerCast output dtype: {layercast_output.dtype}")
    print(f"LayerCast output shape: {layercast_output.shape}")
    print(f"LayerCast output (first 5 values): {layercast_output[0, :5]}")
    
    # 检查输出是否仍然是 BF16
    assert layercast_output.dtype == torch.bfloat16, f"Expected BF16, got {layercast_output.dtype}"
    print("\n✅ Output dtype is correct (BF16)")
    
    # 检查输出是否不同（因为计算精度不同）
    diff = torch.abs(original_output - layercast_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nDifference between original and LayerCast:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    # 输出应该不同（因为计算精度不同），但差异应该很小
    assert max_diff > 0, "Outputs should be different due to precision difference"
    print("✅ Outputs are different (as expected due to precision difference)")
    
    # 恢复原始 forward
    remove_layercast(model, original_forwards)
    
    # 验证恢复
    with torch.no_grad():
        restored_output = model(input_tensor)
    
    diff_restored = torch.abs(original_output - restored_output).max().item()
    print(f"\nDifference after restore: {diff_restored:.10f}")
    assert diff_restored < 1e-6, "Model should be restored to original state"
    print("✅ Model successfully restored to original state")
    
    print("\n" + "="*60)
    print("✅ All basic tests passed!")
    print("="*60)


def test_layercast_precision():
    """测试 LayerCast 是否真的使用 FP32 进行计算"""
    print("\nTesting LayerCast precision...")
    
    # 创建一个简单的线性层
    linear = nn.Linear(5, 3, bias=True)
    linear = linear.to(torch.bfloat16)
    
    # 创建测试输入
    input_tensor = torch.randn(2, 5, dtype=torch.bfloat16)
    
    # 手动 FP32 计算（参考）
    weight_fp32 = linear.weight.to(torch.float32)
    bias_fp32 = linear.bias.to(torch.float32) if linear.bias is not None else None
    input_fp32 = input_tensor.to(torch.float32)
    reference_output = torch.nn.functional.linear(input_fp32, weight_fp32, bias_fp32)
    
    # 应用 LayerCast
    original_forwards = apply_layercast(linear)
    
    # LayerCast 输出
    with torch.no_grad():
        layercast_output = linear(input_tensor)
    
    # 转换为 FP32 进行比较
    layercast_output_fp32 = layercast_output.to(torch.float32)
    
    # 比较（应该非常接近，因为都是 FP32 计算）
    diff = torch.abs(reference_output - layercast_output_fp32)
    max_diff = diff.max().item()
    
    print(f"Max difference from reference FP32 computation: {max_diff:.10f}")
    
    # 差异应该非常小（只是转换误差）
    assert max_diff < 1e-5, f"LayerCast should match FP32 computation, but got diff={max_diff}"
    print("✅ LayerCast matches FP32 computation")
    
    # 恢复
    remove_layercast(linear, original_forwards)
    
    print("✅ Precision test passed!")


if __name__ == "__main__":
    print("="*60)
    print("LayerCast Logic Test")
    print("="*60)
    
    try:
        test_layercast_basic()
        test_layercast_precision()
        
        print("\n" + "="*60)
        print("🎉 All tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


