"""
隐式-显式扩散模型训练脚本
用于LD2011_2014数据集的时间序列插补任务

该脚本支持：
1. 不同缺失率的训练（20%-80%）
2. 模型检查点保存和恢复
3. 训练损失曲线记录
4. 多GPU支持
5. 自动清理之前的训练结果

作者: Claude AI
日期: 2025-11-19
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# 导入工具函数
from utils.util import (
    find_max_epoch,
    print_size,
    training_loss,
    calc_diffusion_hyperparams
)

# 导入模型
from imputers.ImplicitExplicitDiffusion import ImplicitExplicitDiffusion


def create_random_mask(data, missing_ratio):
    """
    创建随机缺失掩码 (Random Missing)

    参数:
        data: 输入数据，形状 [batch, channels, length]
        missing_ratio: 缺失率，范围 [0, 1]

    返回:
        mask: 掩码张量，1表示观测，0表示缺失
    """
    # 创建全1掩码
    mask = torch.ones_like(data)

    # 对每个样本和每个通道独立创建掩码
    batch_size, channels, length = data.shape

    for b in range(batch_size):
        for c in range(channels):
            # 计算需要缺失的点数
            n_missing = int(length * missing_ratio)

            # 随机选择缺失位置
            missing_indices = torch.randperm(length)[:n_missing]

            # 设置掩码
            mask[b, c, missing_indices] = 0

    return mask


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          only_generate_missing,
          missing_ratio,
          batch_size,
          device):
    """
    训练扩散模型

    参数说明:
    =========
    output_directory (str): 模型检查点保存路径
    ckpt_iter (int or 'max'): 预训练检查点迭代数
                              'max' 自动选择最新检查点
    n_iters (int): 训练迭代次数
                   建议值: 30000-100000
    iters_per_ckpt (int): 每多少次迭代保存一次检查点
                         建议值: 1000-5000
    iters_per_logging (int): 每多少次迭代记录一次日志
                            建议值: 100-500
    learning_rate (float): 学习率
                          建议值: 1e-4 到 5e-4
                          - 更大的学习率训练更快但可能不稳定
                          - 更小的学习率更稳定但训练慢
    only_generate_missing (int): 是否只对缺失部分应用扩散
                                1: 只对缺失部分 (推荐)
                                0: 对整个序列
    missing_ratio (float): 数据缺失率
                          范围: 0.0-1.0
                          例如: 0.2 表示20%缺失
    batch_size (int): 批次大小
                     建议值: 8-32
                     - 更大的batch_size训练更稳定但需要更多显存
    device (str): 训练设备
                 'cuda:0' 或 'cuda:1'
    """

    # ===== 1. 设置实验路径 =====
    # 根据扩散超参数创建实验目录名称
    local_path = "T{}_beta0{}_betaT{}_missing{}".format(
        diffusion_config["T"],
        diffusion_config["beta_0"],
        diffusion_config["beta_T"],
        int(missing_ratio * 100)  # 转换为百分比
    )

    # 创建完整输出路径
    output_directory = os.path.join(output_directory, local_path)

    # 如果目录已存在，清空之前的训练结果
    if os.path.exists(output_directory):
        print(f"检测到已存在的实验目录: {output_directory}")
        print("清空之前的训练结果...")
        shutil.rmtree(output_directory)

    # 创建新目录
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    print(f"实验输出目录: {output_directory}")

    # 创建loss记录目录
    loss_dir = os.path.join(output_directory, 'losses')
    os.makedirs(loss_dir, exist_ok=True)

    # ===== 2. 设置设备 =====
    # 将扩散超参数移到指定GPU
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # ===== 3. 创建模型 =====
    print("\n创建隐式-显式扩散模型...")
    net = ImplicitExplicitDiffusion(**model_config).to(device)
    print_size(net)

    # ===== 4. 创建优化器 =====
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # ===== 5. 加载检查点（如果存在）=====
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)

    if ckpt_iter >= 0:
        try:
            # 加载检查点文件
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location=device)

            # 加载模型和优化器状态
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f'成功加载检查点，迭代数: {ckpt_iter}')
        except Exception as e:
            ckpt_iter = -1
            print(f'加载检查点失败: {e}')
            print('从头开始训练')
    else:
        ckpt_iter = -1
        print('从头开始训练')

    # ===== 6. 加载训练数据 =====
    print("\n加载训练数据...")
    training_data = np.load(trainset_config['train_data_path'])
    print(f"原始数据形状: {training_data.shape}")

    # 数据形状: [样本数, 序列长度, 特征数]
    # 需要转换为: [样本数, 特征数, 序列长度]
    training_data = torch.from_numpy(training_data).float()
    training_data = training_data.permute(0, 2, 1)  # [N, C, L]

    # 创建数据加载器
    # 将数据分成若干批次
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(training_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 随机打乱
        drop_last=True  # 丢弃最后不完整的批次
    )

    print(f"训练数据形状: {training_data.shape}")
    print(f"批次数: {len(train_loader)}")

    # ===== 7. 训练循环 =====
    print("\n开始训练...")
    print(f"总迭代次数: {n_iters}")
    print(f"缺失率: {missing_ratio * 100}%")

    # 用于记录损失
    loss_history = []
    iter_history = []

    n_iter = ckpt_iter + 1
    epoch = 0

    # 设置模型为训练模式
    net.train()

    # 进度条
    pbar = tqdm(total=n_iters - n_iter, desc="训练进度")

    while n_iter < n_iters + 1:
        epoch += 1

        for batch_data in train_loader:
            # 获取批次数据
            batch = batch_data[0].to(device)  # [batch_size, channels, length]

            # 创建随机缺失掩码
            mask = create_random_mask(batch, missing_ratio)
            mask = mask.to(device)

            # 创建损失掩码（与mask相反）
            loss_mask = ~mask.bool()

            # 确保维度一致
            assert batch.size() == mask.size() == loss_mask.size()

            # ===== 前向传播和反向传播 =====
            optimizer.zero_grad()

            # 准备输入: (音频数据, 条件数据, 掩码, 损失掩码)
            X = (batch, batch, mask, loss_mask)

            # 计算训练损失
            loss = training_loss(
                net,
                nn.MSELoss(),
                X,
                diffusion_hyperparams,
                only_generate_missing=only_generate_missing
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            # ===== 记录日志 =====
            if n_iter % iters_per_logging == 0:
                loss_value = loss.item()
                loss_history.append(loss_value)
                iter_history.append(n_iter)

                tqdm.write(f"迭代 {n_iter}/{n_iters} | Epoch {epoch} | "
                          f"损失: {loss_value:.6f}")

                # 保存损失历史
                np.save(
                    os.path.join(loss_dir, 'loss_history.npy'),
                    np.array(loss_history)
                )
                np.save(
                    os.path.join(loss_dir, 'iter_history.npy'),
                    np.array(iter_history)
                )

            # ===== 保存检查点 =====
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                checkpoint_path = os.path.join(output_directory, checkpoint_name)

                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': n_iter,
                    'loss': loss.item()
                }, checkpoint_path)

                tqdm.write(f'检查点已保存: {checkpoint_path}')

            n_iter += 1
            pbar.update(1)

            # 达到最大迭代次数则停止
            if n_iter >= n_iters + 1:
                break

    pbar.close()

    # ===== 8. 绘制训练损失曲线 =====
    print("\n绘制训练损失曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(iter_history, loss_history, linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('训练损失', fontsize=12)
    plt.title(f'训练损失曲线 (缺失率: {missing_ratio*100}%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    loss_curve_path = os.path.join(output_directory, 'training_loss_curve.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存: {loss_curve_path}")

    plt.close()

    print("\n训练完成！")


if __name__ == "__main__":
    """
    主函数

    使用方法:
    python train_ld2011.py -c config/config_ImplicitExplicit_LD2011.json --missing_ratio 0.2

    参数:
    -c: 配置文件路径
    --missing_ratio: 缺失率 (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    --device: GPU设备 (cuda:0 或 cuda:1)
    """
    parser = argparse.ArgumentParser(description='训练隐式-显式扩散模型')

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/config_ImplicitExplicit_LD2011.json',
        help='配置文件路径'
    )

    parser.add_argument(
        '--missing_ratio',
        type=float,
        default=None,
        help='缺失率 (0.0-1.0)，例如0.2表示20%缺失'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='训练设备 (cuda:0 或 cuda:1)'
    )

    args = parser.parse_args()

    # ===== 加载配置文件 =====
    with open(args.config) as f:
        config = json.loads(f.read())

    print("=" * 80)
    print("隐式-显式扩散模型 - 训练脚本")
    print("=" * 80)
    print("\n配置信息:")
    print(json.dumps(config, indent=2))
    print("=" * 80)

    # 提取各部分配置
    train_config = config["train_config"]
    global trainset_config
    trainset_config = config["trainset_config"]
    global diffusion_config
    diffusion_config = config["diffusion_config"]
    global model_config
    model_config = config["model_config"]

    # 计算扩散超参数
    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    # 覆盖命令行参数
    if args.missing_ratio is not None:
        train_config['missing_ratio'] = args.missing_ratio
        print(f"\n使用命令行指定的缺失率: {args.missing_ratio * 100}%")

    if args.device is not None:
        train_config['device'] = args.device
        print(f"使用命令行指定的设备: {args.device}")

    # 开始训练
    train(**train_config)
