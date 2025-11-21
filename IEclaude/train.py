"""
IEclaude训练脚本 (Training Script)

功能：
    训练隐式显式扩散模型用于交通数据插补

使用方法：
    python train.py --config configs/config_20.json --gpu 0

超参数调优指南：
    见各个模块的注释和README文档
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil

# 导入模块
from models import IEDiffusionModel
from data import load_traffic_data, create_dataloader
from utils import DiffusionProcess, DiffusionLoss


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='训练IEclaude模型')

    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (JSON格式)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU编号 (0 for A40, 1 for A10)')
    parser.add_argument('--resume', type=str, default=None,
                       help='从checkpoint恢复训练')

    return parser.parse_args()


def load_config(config_path):
    """
    加载配置文件

    参数：
        config_path (str): 配置文件路径

    返回：
        config (dict): 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def set_seed(seed=42):
    """
    设置随机种子以保证可复现性

    参数：
        seed (int): 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(model, optimizer, epoch, loss, save_path, is_best=False):
    """
    保存checkpoint

    参数：
        model (nn.Module): 模型
        optimizer: 优化器
        epoch (int): 当前epoch
        loss (float): 当前损失
        save_path (str): 保存路径
        is_best (bool): 是否是最佳模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
        shutil.copyfile(save_path, best_path)


def train_epoch(model, dataloader, diffusion_process, loss_fn, optimizer, device, epoch, total_epochs):
    """
    训练一个epoch

    参数：
        model: 模型
        dataloader: 数据加载器
        diffusion_process: 扩散过程
        loss_fn: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        total_epochs: 总epoch数

    返回：
        avg_loss (float): 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}')

    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        observed_data = batch['observed_data'].to(device)  # [B, C, L]
        mask = batch['mask'].to(device)  # [B, C, L]
        ground_truth = batch['ground_truth'].to(device)  # [B, C, L]

        B, C, L = observed_data.shape

        # === 前向扩散：添加噪声 ===
        # 随机采样扩散步骤
        t = torch.randint(0, diffusion_process.T, (B,), device=device)

        # 添加噪声得到x_t
        x_t, true_noise = diffusion_process.q_sample(ground_truth, t)

        # === 模型预测噪声 ===
        predicted_noise = model(x_t, observed_data, mask, t)

        # === 计算损失 ===
        loss = loss_fn(predicted_noise, true_noise, mask)

        # === 反向传播和优化 ===
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 累积损失
        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    # 计算平均损失
    avg_loss = total_loss / num_batches

    return avg_loss


def train(config, args):
    """
    主训练函数

    参数：
        config (dict): 配置字典
        args: 命令行参数
    """
    print("=" * 80)
    print("IEclaude 训练脚本")
    print("=" * 80)

    # 设置随机种子
    set_seed(config.get('seed', 42))

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建输出目录
    output_dir = config['train']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 清除之前的训练结果
    if config['train'].get('clean_before_train', True):
        print(f"\n清除之前的训练结果...")
        for file in os.listdir(output_dir):
            if file.endswith('.png') or file.endswith('.pt') or file.endswith('.txt'):
                os.remove(os.path.join(output_dir, file))

    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. 加载数据")
    print("=" * 80)

    train_data, test_data, scaler, data_info = load_traffic_data(
        data_path=config['data']['data_path'],
        seq_len=config['data']['seq_len'],
        stride=config['data'].get('stride', config['data']['seq_len'] // 2),
        train_ratio=config['data'].get('train_ratio', 0.7),
        normalize_method=config['data'].get('normalize', 'standard'),
        verbose=True
    )

    # 创建数据加载器
    train_loader = create_dataloader(
        data=train_data,
        missing_rate=config['train']['missing_rate'],
        missing_pattern=config['train'].get('missing_pattern', 'random'),
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train'].get('num_workers', 0)
    )

    print(f"\n训练集batch数量: {len(train_loader)}")

    # ========================================================================
    # 2. 创建模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 创建模型")
    print("=" * 80)

    model = IEDiffusionModel(
        in_channels=data_info['num_channels'],
        res_channels=config['model']['res_channels'],
        skip_channels=config['model']['skip_channels'],
        out_channels=data_info['num_channels'],
        num_res_layers=config['model']['num_res_layers'],
        dilation_cycle=config['model'].get('dilation_cycle', 10),
        # 扩散步骤嵌入
        diffusion_step_embed_dim_in=config['diffusion']['embed_dim_in'],
        diffusion_step_embed_dim_mid=config['diffusion']['embed_dim_mid'],
        diffusion_step_embed_dim_out=config['diffusion']['embed_dim_out'],
        # TCN参数
        tcn_channels=config['model']['tcn_channels'],
        tcn_kernel_size=config['model']['tcn_kernel_size'],
        tcn_dilation_rates=config['model']['tcn_dilation_rates'],
        tcn_dropout=config['model'].get('tcn_dropout', 0.0),
        # S4参数
        s4_d_state=config['model']['s4_d_state'],
        s4_n_layers=config['model']['s4_n_layers'],
        s4_l_max=config['data']['seq_len'],
        s4_dropout=config['model'].get('s4_dropout', 0.0),
        s4_bidirectional=config['model'].get('s4_bidirectional', True)
    ).to(device)

    # 打印模型信息
    model_config = model.get_config()
    print(f"\n模型配置:")
    print(f"  输入通道数: {model_config['in_channels']}")
    print(f"  残差通道数: {model_config['res_channels']}")
    print(f"  残差层数: {model_config['num_res_layers']}")
    print(f"  TCN感受野: {model_config['tcn_config']['receptive_field']} 时间步")
    print(f"  S4状态维度: {model_config['s4_config']['d_state']}")
    print(f"  S4层数: {model_config['s4_config']['n_layers']}")
    print(f"  参数总数: {model_config['num_parameters']:,}")

    # ========================================================================
    # 3. 创建扩散过程
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. 创建扩散过程")
    print("=" * 80)

    diffusion_process = DiffusionProcess(
        T=config['diffusion']['T'],
        beta_0=config['diffusion']['beta_0'],
        beta_T=config['diffusion']['beta_T'],
        schedule=config['diffusion'].get('schedule', 'linear')
    )

    print(f"  扩散步数: {diffusion_process.T}")
    print(f"  起始噪声: {diffusion_process.beta_0}")
    print(f"  结束噪声: {diffusion_process.beta_T}")
    print(f"  调度方式: {diffusion_process.schedule}")

    # 创建损失函数
    loss_fn = DiffusionLoss(
        only_generate_missing=config['train'].get('only_generate_missing', True)
    )

    # ========================================================================
    # 4. 创建优化器和调度器
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 创建优化器")
    print("=" * 80)

    optimizer = Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train'].get('weight_decay', 0.0)
    )

    # 学习率调度器
    scheduler_type = config['train'].get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs'],
            eta_min=config['train'].get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config['train'].get('step_size', 50),
            gamma=config['train'].get('gamma', 0.5)
        )
    else:
        scheduler = None

    print(f"  优化器: Adam")
    print(f"  初始学习率: {config['train']['learning_rate']}")
    print(f"  调度器: {scheduler_type}")

    # ========================================================================
    # 5. 训练循环
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. 开始训练")
    print("=" * 80)

    num_epochs = config['train']['epochs']
    save_interval = config['train'].get('save_interval', 10)

    train_losses = []
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # 训练一个epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            diffusion_process=diffusion_process,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=num_epochs
        )

        train_losses.append(avg_loss)

        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config['train']['learning_rate']

        # 打印信息
        print(f"\nEpoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.6f}")

        # 保存checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
            print(f"  已保存checkpoint: {checkpoint_path}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, avg_loss, best_path, is_best=True)
            print(f"  已保存最佳模型: {best_path}")

    # ========================================================================
    # 6. 保存训练曲线
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. 保存训练曲线")
    print("=" * 80)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  训练曲线已保存: {os.path.join(output_dir, 'training_loss.png')}")

    # 保存损失值
    loss_file = os.path.join(output_dir, 'training_losses.txt')
    with open(loss_file, 'w') as f:
        f.write('Epoch,Loss\n')
        for epoch, loss in enumerate(train_losses, 1):
            f.write(f'{epoch},{loss:.6f}\n')

    print(f"  损失值已保存: {loss_file}")

    # ========================================================================
    # 7. 完成
    # ========================================================================
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型和结果已保存到: {output_dir}")


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 开始训练
    train(config, args)
