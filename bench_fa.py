import torch
from flash_attn import flash_attn_varlen_qkvpacked_func

def benchmark_flash_attn_fixed_len():
    # ================= 配置参数 =================
    BATCH_SIZE = 16          # 批次大小
    SEQ_LEN = 32 * 1024         # 固定序列长度
    HIDDEN_SIZE = 5120       # 隐藏层维度
    NUM_HEADS = 40           # 头数
    
    # Head Dim = 4096 / 40 = 102 (向下取整)
    # 警告: 102 不是 2 的幂次，也不是 32/64/128 的倍数。
    # FlashAttn 会自动 pad 到 128 计算，这会导致有效 TFLOPS 低于硬件峰值。
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS 
    
    DTYPE = torch.bfloat16
    DEVICE = "cuda"
    IS_CAUSAL = True
    
    # ================= 构造固定长度数据 =================
    total_tokens = BATCH_SIZE * SEQ_LEN
    
    # 构造 cu_seqlens: [0, 4096, 8192, 12288, ...]
    # 形状为 (BATCH_SIZE + 1)
    cu_seqlens = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN, step=SEQ_LEN, 
        device=DEVICE, dtype=torch.int32
    )
    
    max_seqlen = SEQ_LEN

    print(f"--- Configuration ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Seq Len (Fixed): {SEQ_LEN}")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"Num Heads: {NUM_HEADS}")
    print(f"Head Dim: {HEAD_DIM} (Note: Padded to 128 internally)")
    print(f"Total Tokens: {total_tokens}")
    print(f"Causal: {IS_CAUSAL}")
    print(f"---------------------")

    # ================= 准备输入 =================
    # varlen 接口要求输入为 (total_tokens, 3, n_heads, head_dim)
    qkv = torch.randn(total_tokens, 3, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE).requires_grad_(False)

    # ================= 预热 =================
    print("Warming up...")
    for _ in range(10):
        flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0, causal=IS_CAUSAL
        )
    torch.cuda.synchronize()

    # ================= 性能测试 =================
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iterations = 100
    print(f"Running {iterations} iterations...")
    
    start_event.record()
    for _ in range(iterations):
        flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0, causal=IS_CAUSAL
        )
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event) / iterations
    
    # ================= TFLOPS 计算 =================
    # FLOPs 公式 (Causal): 2 * B * S^2 * H * D
    # 因为 S 是固定的，所以不需要 sum(s_i^2)，直接用 B * S^2 即可
    
    flops_per_iter = 2 * BATCH_SIZE * (SEQ_LEN ** 2) * NUM_HEADS * HEAD_DIM
    
    if not IS_CAUSAL:
        flops_per_iter *= 2
        
    tflops = flops_per_iter / (elapsed_time_ms / 1000) / 1e12

    print(f"\nResults:")
    print(f"Avg Latency: {elapsed_time_ms:.3f} ms")
    print(f"Theoretical FLOPs: {flops_per_iter:.3e}")
    print(f"Throughput: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    benchmark_flash_attn_fixed_len()