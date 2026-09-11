import os
import torch

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