import os
import unittest
import torch

class TestMamba(unittest.TestCase):
    
    def setUp(self):
        ...
        
    def test01(self):
        test_selective_scan()
    
    
def test_selective_scan():
    print("=" * 60)
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    print(f"显卡算力架构: {torch.cuda.get_device_capability(0)}")
    print("=" * 60)

    # 1. 验证 Causal-Conv1d 算子
    print("[1/4] 测试 Causal-Conv1d 底层算子...")
    from causal_conv1d import causal_conv1d_fn
    x = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    conv_weight = torch.randn(64, 4, device="cuda", dtype=torch.float16)
    out_conv = causal_conv1d_fn(x, conv_weight)
    print("  -> Causal-Conv1d 运行成功！形状:", out_conv.shape)

    # 2. 验证 Mamba 1 (selective_scan_fn C++ 扩展)
    print("[2/4] 测试 Mamba 1 (selective_scan_fn)...")
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    u = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    delta = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    A = torch.randn(64, 16, device="cuda", dtype=torch.float32)
    B = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    C = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
    out_ssm = selective_scan_fn(u, delta, A, B, C)
    print("  -> selective_scan_fn 执行成功！形状:", out_ssm.shape)

    # 3. 验证 Mamba 2 (Triton SSD 算子)
    print("[3/4] 测试 Mamba 2 (Triton State Space Duality)...")
    from mamba_ssm.modules.mamba2 import Mamba2
    layer_m2 = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2).to("cuda").to(torch.bfloat16)
    tokens = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    out_m2 = layer_m2(tokens)
    print("  -> Mamba 2 Triton 核心前向传播成功！形状:", out_m2.shape)

    # 4. 验证 MambaVision 视觉模型
    # print("[4/4] 测试 MambaVision 模型前向推断...")
    # from mambavision.models.mamba_vision import mamba_vision_T
    # model = mamba_vision_T(pretrained=False).to("cuda")
    # img = torch.randn(1, 3, 224, 224, device="cuda")
    # with torch.no_grad():
    #     logits = model(img)
    # print("  -> MambaVision-T 前向通过！输出维度:", logits.shape)
    # print("=" * 60)
    # print("恭喜！所有 Mamba 模块、Triton JIT 及 C++ CUDA 加速库已在 RTX 5070 Ti 上正常运行！")

if __name__ == "__main__":
    test_selective_scan()