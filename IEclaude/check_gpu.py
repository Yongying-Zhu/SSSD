#!/usr/bin/env python3
"""
GPU检测脚本

检查系统中可用的GPU设备
"""

import torch

print("=" * 80)
print("GPU 设备检测")
print("=" * 80)

# 检查CUDA是否可用
if torch.cuda.is_available():
    print(f"\n✓ CUDA可用")
    print(f"  CUDA版本: {torch.version.cuda}")

    # 获取GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"  可用GPU数量: {num_gpus}")

    # 列出所有GPU
    print(f"\nGPU详细信息:")
    for i in range(num_gpus):
        print(f"\n  GPU {i}:")
        print(f"    名称: {torch.cuda.get_device_name(i)}")

        # 获取显存信息
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3  # 转换为GB
        print(f"    显存: {total_memory:.2f} GB")

        # 当前显存使用情况
        if i < num_gpus:
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    已分配: {allocated:.2f} GB")
            print(f"    已保留: {reserved:.2f} GB")

    # 使用建议
    print(f"\n" + "=" * 80)
    print("使用建议:")
    print("=" * 80)

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3

        print(f"\nGPU {i} ({gpu_name}):")
        print(f"  训练命令: python train.py --config configs/config_20.json --gpu {i}")
        print(f"  或使用脚本: ./run_all.sh --gpu {i}")

        # 根据显存给出batch_size建议
        if total_memory < 12:
            print(f"  建议batch_size: 4 (显存较小)")
        elif total_memory < 24:
            print(f"  建议batch_size: 8 (默认)")
        else:
            print(f"  建议batch_size: 16 (显存充足)")

else:
    print(f"\n✗ CUDA不可用")
    print(f"  将使用CPU进行训练（速度会很慢）")
    print(f"  训练命令: python train.py --config configs/config_20.json")

print("\n" + "=" * 80)
