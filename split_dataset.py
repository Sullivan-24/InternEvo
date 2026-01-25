import os
import shutil
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 原 1TB .bin 文件路径 (请修改这里)
SOURCE_BIN_PATH = "/mnt/shared-storage-user/lusitian/data/data_jsonl/github/tokenized_llama2/train_folder/data/output.bin" 

# 输出目录 (将在此目录下创建 10 个子文件夹)
OUTPUT_DIR = "/mnt/shared-storage-user/ailab-sys/lusitian/data/github_split/"

# 配置参数
SPLIT_NUM = 10     # 将前 10% 切分成 10 个文件
TAKE_RATIO = 0.1   # 只取数据集的前 10%
CHUNK_SIZE = 128 * 1024 * 1024  # IO buffer 128MB
# ===========================================

def split_jsonl_bin_dataset():
    meta_path = SOURCE_BIN_PATH + ".meta"
    
    if not os.path.exists(SOURCE_BIN_PATH) or not os.path.exists(meta_path):
        print(f"Error: Source file or meta file not found: {SOURCE_BIN_PATH}")
        return

    # 1. 读取 Meta 信息
    print(f"Loading metadata from {meta_path}...")
    # shape: [N, 2], dtype通常是 int64 或 int32
    full_meta = np.load(meta_path, mmap_mode='r') 
    
    total_samples = len(full_meta)
    file_size_bytes = os.path.getsize(SOURCE_BIN_PATH)
    
    # 2. 确定切分样本数
    target_samples = int(total_samples * TAKE_RATIO)
    samples_per_shard = target_samples // SPLIT_NUM
    
    print(f"Total Samples: {total_samples}")
    print(f"File Size: {file_size_bytes / (1024**3):.2f} GB")
    print(f"Target Samples (Top {TAKE_RATIO*100}%): {target_samples}")
    print(f"Splitting into {SPLIT_NUM} folders, approx {samples_per_shard} samples each.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(SOURCE_BIN_PATH, "rb") as src_f:
        for i in range(SPLIT_NUM):
            # --- A. 确定样本索引范围 ---
            idx_start = i * samples_per_shard
            idx_end = (i + 1) * samples_per_shard
            
            if i == SPLIT_NUM - 1:
                idx_end = target_samples # 最后一个文件补齐到 target_samples
            
            # --- B. 提取 Meta 片段 ---
            shard_meta = np.array(full_meta[idx_start : idx_end])
            
            if len(shard_meta) == 0:
                continue
                
            # --- C. 计算字节范围 ---
            start_byte_offset = int(shard_meta[0][0])
            
            # 计算结束字节位置
            if idx_end < total_samples:
                end_byte_offset = int(full_meta[idx_end][0])
            else:
                end_byte_offset = file_size_bytes
            
            byte_length = end_byte_offset - start_byte_offset
            
            # --- D. 构建文件名和独立的子文件夹 ---
            # 1. 创建子文件夹，例如: /.../github_split/part_000/
            sub_folder_name = f"part_{i:03d}"
            out_dir_shard = os.path.join(OUTPUT_DIR, sub_folder_name)
            os.makedirs(out_dir_shard, exist_ok=True)
            
            # 2. 构建在新文件夹中的文件路径，例如: /.../part_000/output.bin
            base_name = "output" 
            out_bin_path = os.path.join(out_dir_shard, f"{base_name}.bin")
            out_meta_path = out_bin_path + ".meta"
            
            print(f"\n[{i+1}/{SPLIT_NUM}] Writing to folder: {out_dir_shard}")
            print(f"  Sample Range: {idx_start} -> {idx_end} ({len(shard_meta)} samples)")
            print(f"  Byte Range:   {start_byte_offset} -> {end_byte_offset}")
            
            # --- E. 写 .bin 文件 ---
            src_f.seek(start_byte_offset)
            with open(out_bin_path, "wb") as dst_f:
                remaining = byte_length
                pbar = tqdm(total=byte_length, unit='B', unit_scale=True, leave=False)
                while remaining > 0:
                    read_size = min(remaining, CHUNK_SIZE)
                    data = src_f.read(read_size)
                    dst_f.write(data)
                    remaining -= read_size
                    pbar.update(read_size)
                pbar.close()
                
            # --- F. 写 .meta 文件 ---
            # 修正 Offset
            shard_meta[:, 0] -= start_byte_offset
            
            # 保存 meta (numpy 自动加 .npy)
            np.save(out_meta_path, shard_meta)
            
            # 重命名移除 .npy 后缀 (如果必须)
            if os.path.exists(out_meta_path + ".npy"):
                os.rename(out_meta_path + ".npy", out_meta_path)

    print(f"\nDone! Output directory: {OUTPUT_DIR}")
    print("Action: Update your 'TRAIN_FOLDER' config to this directory.")

if __name__ == "__main__":
    split_jsonl_bin_dataset()