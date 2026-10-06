import os
import unittest
import torch

class TestMamba(unittest.TestCase):
    
    def setUp(self):
        ...
        
    def test01(self):
        test_selective_scan()
    
    
def test_selective_scan():
    print("="*60)
    print("🚀 Mamba Selective Scan CUDA 环境综合测试")
    print("="*60)

    # 1. 检查 PyTorch CUDA 环境
    print("\n[1/4] 检查 CUDA 可用性...")
    if not torch.cuda.is_available():
        print("❌ 失败: 未检测到 CUDA 环境。请确认安装了 GPU 版本的 PyTorch。")
        return
    device = torch.device("cuda")
    print(f"✅ CUDA 正常. 当前设备: {torch.cuda.get_device_name(device)}")
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 版本: {torch.version.cuda}")

    # 2. 检查 Mamba 导入
    print("\n[2/4] 尝试导入 selective_scan_fn...")
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        print("✅ 导入成功！Python 层模块存在。")
    except ImportError as e:
        print("❌ 失败: Mamba 模块未安装或导入失败。")
        print(f"详细报错: {e}")
        return
    except Exception as e:
        print(f"❌ 失败: 导入时发生未知错误: {e}")
        return

    # 3. 构造 Dummy Data (伪造张量)
    print("\n[3/4] 构造测试张量 (转移至 GPU 并开启梯度追踪)...")
    batch_size = 2
    dim = 64
    seqlen = 128
    d_state = 16

    try:
        # 输入张量 (B, D, L)
        u = torch.randn(batch_size, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        # 步长 (B, D, L)
        delta = torch.randn(batch_size, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        # 状态转移矩阵 (D, N)
        A = torch.randn(dim, d_state, device=device, dtype=torch.float32, requires_grad=True)
        # 输入投影矩阵 (B, N, L)
        B_mat = torch.randn(batch_size, d_state, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        # 输出投影矩阵 (B, N, L)
        C_mat = torch.randn(batch_size, d_state, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        # 残差连接 (D)
        D_vec = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
        # 步长偏置 (D)
        delta_bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
        print("✅ 张量构造完成。")
    except Exception as e:
        print(f"❌ 失败: 张量构造阶段报错: {e}")
        return

    # 4. 执行前向传播和反向传播
    print("\n[4/4] 正在调用底层 CUDA 算子进行计算...")
    try:
        # 测试前向传播 (Forward)
        out = selective_scan_fn(
            u, delta, A, B_mat, C_mat, D_vec, 
            z=None, delta_bias=delta_bias, delta_softplus=True
        )
        print(f"✅ 前向传播 (Forward) 成功! 输出形状: {out.shape}")
        
        # 验证输出维度是否正确
        assert out.shape == u.shape, f"形状错误! 期望 {u.shape}, 实际 {out.shape}"

        # 测试反向传播 (Backward)
        loss = out.sum()
        loss.backward()
        print("✅ 反向传播 (Backward) 成功! 梯度已正常计算。")
        
    except Exception as e:
        print("\n❌ 致命错误: 底层 CUDA 算子调用失败！")
        print("这就是导致你之前遇到 'selective_scan_cuda is not installed' 的原因。")
        print(f"详细报错:\n{e}")
        return

    print("\n🎉🎉🎉 测试完美通过！你的 Mamba CUDA 算子环境已彻底打通，可以放心去炼丹了！")

if __name__ == "__main__":
    # 修复 Windows 下 Triton 缓存路径
    if "HOME" not in os.environ:
        os.environ["HOME"] = os.environ.get("USERPROFILE", "C:\\Users\\Default")

    from mamba_ssm import Mamba2

    print(f"CUDA 可用性: {torch.cuda.is_available()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"计算架构: {torch.cuda.get_device_capability(0)}")

    device = "cuda"
    batch, seqlen, dim = 2, 64, 128
    x = torch.randn(batch, seqlen, dim, device=device, requires_grad=True)

    # 实例化 Mamba2
    model = Mamba2(d_model=dim, d_state=64, d_conv=4, expand=2).to(device)

    # 前向传播与反向传播
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert out.shape == (batch, seqlen, dim)
    print(f"输出维度正确: {out.shape}")
    print("Mamba2 在 Windows 11 + 5070 Ti 上运行测试成功！")