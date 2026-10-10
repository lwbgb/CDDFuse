import gc
import sys
import torch
import torch.nn as nn

# 检查 CUDA 与显卡环境
if not torch.cuda.is_available():
    raise SystemError("未检测到可用的 CUDA 设备！")

device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
compute_cap = torch.cuda.get_device_capability(0)

# 动态加载 Mamba 各代模块
models_dict = {}

try:
    from mamba_ssm import Mamba
    models_dict["Mamba-1"] = lambda d_model: Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
except Exception as e:
    print(f"[警告] Mamba-1 加载失败: {e}")

try:
    from mamba_ssm import Mamba2
    models_dict["Mamba-2"] = lambda d_model: Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64)
except Exception as e:
    print(f"[警告] Mamba-2 加载失败: {e}")

try:
    from mamba_ssm.modules.mamba3 import Mamba3
    models_dict["Mamba-3"] = lambda d_model: Mamba3(d_model=d_model)
except Exception as e:
    print(f"[警告] Mamba-3 加载跳过: {e}")


def benchmark_forward(model: nn.Module, x: torch.Tensor, warmup_steps: int = 15, test_steps: int = 50):
    """测量单层模块的前向推理延迟与峰值显存"""
    model.eval()

    # 1. 预热 GPU 与 JIT / Triton 编译内核，稳定时钟频率
    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = model(x)
    torch.cuda.synchronize()

    # 2. 清理缓存并测量显存峰值
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    with torch.inference_mode():
        for _ in range(test_steps):
            start_event.record()
            out = model(x)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))  # 单位：毫秒 (ms)

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_latency_ms = sum(latencies) / len(latencies)
    batch_size, seq_len, _ = x.shape
    tokens_per_sec = (batch_size * seq_len) / (avg_latency_ms / 1000.0)

    return avg_latency_ms, peak_memory_mb, tokens_per_sec


def main():
    print("=" * 82)
    print(f" 设备型号: {gpu_name} | 算力架构: sm_{compute_cap[0]}{compute_cap[1]}")
    print(f" 精度模式: torch.bfloat16 (原生硬件加速模式)")
    print("=" * 82)

    # 测试矩阵配置
    d_model = 1024
    batch_size = 4
    seq_lengths = [1024, 2048, 4096, 8192]

    # 输出表格头
    print(f"{'模型架构':<12} | {'序列长度':<8} | {'Batch':<6} | {'延迟 (ms)':<10} | {'显存峰值 (MB)':<14} | {'吞吐 (tokens/s)':<16}")
    print("-" * 82)

    dtype = torch.bfloat16

    for seq_len in seq_lengths:
        for model_name, model_fn in models_dict.items():
            try:
                # 实例化模型并转移至 GPU
                model = model_fn(d_model).to(device=device, dtype=dtype)
                x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

                latency, peak_mem, throughput = benchmark_forward(model, x)

                print(f"{model_name:<12} | {seq_len:<8} | {batch_size:<6} | {latency:<10.2f} | {peak_mem:<14.2f} | {throughput:<16.1f}")

                del model, x
                gc.collect()
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"{model_name:<12} | {seq_len:<8} | {batch_size:<6} | {'OOM':<10} | {'OOM':<14} | {'-':<16}")
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as err:
                print(f"{model_name:<12} | {seq_len:<8} | {batch_size:<6} | {'ERROR':<10} | {str(err)[:12]:<14} | {'-':<16}")

        print("-" * 82)


if __name__ == "__main__":
    main()