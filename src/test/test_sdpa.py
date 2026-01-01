#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/28 12:54
@project: LucaGenPostTraining
@file: test_sdpa.py
@desc: xxxx
'''
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

def check_sdpa_backend():
    # 1. 检查基本环境
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot use FlashAttention.")
        return

    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        print(f"GPU Compute Capability is {major}.x, which is less than 8.0. FlashAttention may not be supported or optimal.")

    try:
        import flash_attn
        print(f"flash-attn version: {flash_attn.__version__}")
    except ImportError:
        print("flash-attn is not installed. SDPA will not use it.")
        return

    # 2. 准备满足条件的输入
    # 使用 bfloat16 (如果支持) 或 float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # (batch_size, num_heads, seq_len_q, head_dim) -> PyTorch SDPA 推荐的 NHSD 格式
    # FlashAttention 内部会处理格式转换
    query = torch.randn(2, 4, 1024, 64, device="cuda", dtype=dtype)
    key = torch.randn(2, 4, 1024, 64, device="cuda", dtype=dtype)
    value = torch.randn(2, 4, 1024, 64, device="cuda", dtype=dtype)

    print("\n--- Checking SDPA Backend Selection ---")

    # 3. 使用 sdpa_kernel 上下文管理器进行验证
    # 我们只查询，不执行计算
    enabled_backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION, # PyTorch Memory-Efficient
        SDPBackend.MATH               # PyTorch Fallback
    ]

    with sdpa_kernel(enabled_backends):
        # 这个代码块内部会告诉我们，如果现在调用 F.scaled_dot_product_attention，
        # 它会选择哪个后端。
        print(f"For the current inputs (dtype={dtype}, causal=True), SDPA will use:")
        # 我们用一个 try-except 来优雅地捕获错误，并调用 F.scaled_dot_product_attention 来触发后端选择
        try:
            # 传入 is_causal=True, 这是 FlashAttention 的典型用例
            _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            # 如果没有显式打印，我们可以假定它在内部工作，
            # 但更可靠的方式是与 profiler 结合或查看日志
        except Exception as e:
            print(f"An error occurred: {e}")

    print("\nTo be absolutely sure, let's force the backend and see if it works.")

    # 4. 强制使用 FlashAttention 并检查是否报错
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            print("Attempting to force use of FlashAttention backend...")
            output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            print("✅ SUCCESS: FlashAttention backend was successfully used.")
            print(f"Output tensor shape: {output.shape}, dtype: {output.dtype}")
    except Exception as e:
        print(f"❌ FAILED: Could not force FlashAttention. Error: {e}")
        print("This means the conditions for using FlashAttention were not met.")

if __name__ == "__main__":
    check_sdpa_backend()

    import torch
    import torch.nn.functional as F

    # ... (复用方法一中的输入 tensor 定义) ...
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    query = torch.randn(2, 4, 1024, 64, device="cuda", dtype=dtype)
    key = torch.randn(2, 4, 1024, 64, device="cuda", dtype=dtype)
    value = torch.randn(2, 4, 1024, 64, device="cuda", dtype=dtype)

    # 使用 Profiler
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True
    ) as prof:
        with torch.profiler.record_function("sdpa_call"):
            # 在这里调用 SDPA
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    # 打印分析结果
    # 我们关心的是 CUDA Kernel 的执行时间
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))