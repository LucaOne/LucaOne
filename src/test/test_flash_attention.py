#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/28 12:53
@project: LucaGenPostTraining
@file: test_flash_attention.py
@desc: xxxx
'''
import torch
from flash_attn import flash_attn_func

def verify_flash_attention():
    if not torch.cuda.is_available():
        print("CUDA is not available. FlashAttention requires a GPU.")
        return

    # 获取当前GPU的计算能力
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        print(f"GPU compute capability is {major}.{minor}, which is less than 8.0.")
        print("FlashAttention requires Ampere (8.0) or newer architectures.")
        # 注意：FlashAttention v2.1+ 开始支持 7.5 (Turing)，但性能非最优
        # 如果你确实是Turing卡，可以继续尝试，但最好是Ampere以上
        # return

    print("Environment check passed. Running a small test...")

    # bfloat16 is preferred on Ampere and newer GPUs
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    batch_size, seq_len, num_heads, head_dim = 4, 1024, 16, 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)

    try:
        # 使用 flash_attn_func
        output = flash_attn_func(q, k, v, causal=True)
        print("FlashAttention test successful!")
        print(f"Input shape: {q.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == q.shape
    except Exception as e:
        print("FlashAttention test failed.")
        print(e)

if __name__ == "__main__":
    verify_flash_attention()