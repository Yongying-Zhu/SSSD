"""
IEclaude评估脚本 (Evaluation Script)

功能：
    加载训练好的模型，进行插补并评估性能

使用方法：
    python evaluate.py --config configs/config_20.json --checkpoint results/traffic_20/best_model.pt --gpu 0

评估指标：
    - MAE (Mean Absolute Error): 平均绝对误差
    - RMSE (Root Mean Squared Error): 均方根误差
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入模块
from models import IEDiffusionModel
from data import load_traffic_data, create_dataloader
from utils import DiffusionProcess


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估IEclaude模型')

    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU编号')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='每个样本的生成次数（用于平均）')

    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def calculate_metrics(predictions, ground_truth, mask):
    """
    计算评估指标

    参数：
        predictions (np.array): 预测值 [N, C, L]
        ground_truth (np.array): 真实值 [N, C, L]
        mask (np.array): mask [N, C, L], 1=观测, 0=缺失

    返回：
        metrics (dict): 包含MAE和RMSE的字典
    """
    # 只计算缺失位置的指标
    missing_mask = (1 - mask).astype(bool)

    # 提取缺失位置的预测值和真实值
    pred_missing = predictions[missing_mask]
    true_missing = ground_truth[missing_mask]

    # 计算MAE
    mae = np.mean(np.abs(pred_missing - true_missing))

    # 计算RMSE
    mse = np.mean((pred_missing - true_missing) ** 2)
    rmse = np.sqrt(mse)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'num_missing': len(pred_missing)
    }

    return metrics


def impute(model, dataloader, diffusion_process, device, num_samples=10, verbose=True):
    """
    执行插补

    参数：
        model: 模型
        dataloader: 数据加载器
        diffusion_process: 扩散过程
        device: 设备
        num_samples: 每个样本的生成次数
        verbose: 是否显示进度

    返回：
        all_predictions (np.array): 所有预测值
        all_ground_truth (np.array): 所有真实值
        all_masks (np.array): 所有mask
    """
    model.eval()

    all_predictions = []
    all_ground_truth = []
    all_masks = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Imputing') if verbose else dataloader

        for batch in pbar:
            observed_data = batch['observed_data'].to(device)
            mask = batch['mask'].to(device)
            ground_truth = batch['ground_truth'].to(device)

            B, C, L = observed_data.shape

            # 多次采样取平均（降低随机性）
            samples = []
            for _ in range(num_samples):
                # 使用扩散过程生成
                imputed = diffusion_process.p_sample_loop(
                    model=model,
                    shape=(B, C, L),
                    observed_data=observed_data,
                    mask=mask,
                    device=device,
                    verbose=False
                )
                samples.append(imputed.cpu().numpy())

            # 平均多次采样的结果
            imputed_avg = np.mean(samples, axis=0)

            # 合并观测值和插补值
            final = observed_data.cpu().numpy() * mask.cpu().numpy() + \
                    imputed_avg * (1 - mask.cpu().numpy())

            all_predictions.append(final)
            all_ground_truth.append(ground_truth.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    # 拼接所有batch
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    return all_predictions, all_ground_truth, all_masks


def visualize_imputation(predictions, ground_truth, mask, save_path, num_examples=5):
    """
    可视化插补结果

    参数：
        predictions: 预测值 [N, C, L]
        ground_truth: 真实值 [N, C, L]
        mask: mask [N, C, L]
        save_path: 保存路径
        num_examples: 显示的样本数量
    """
    num_examples = min(num_examples, len(predictions))

    fig, axes = plt.subplots(num_examples, 1, figsize=(15, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    for i in range(num_examples):
        # 选择一个通道进行可视化（第一个通道）
        pred = predictions[i, 0, :]
        true = ground_truth[i, 0, :]
        m = mask[i, 0, :]

        # 创建x轴
        x = np.arange(len(pred))

        # 绘制真实值
        axes[i].plot(x, true, 'k-', label='Ground Truth', linewidth=1.5, alpha=0.7)

        # 绘制观测值
        observed_indices = m == 1
        axes[i].scatter(x[observed_indices], true[observed_indices],
                       c='blue', s=20, label='Observed', zorder=3)

        # 绘制插补值
        missing_indices = m == 0
        axes[i].scatter(x[missing_indices], pred[missing_indices],
                       c='red', s=20, label='Imputed', zorder=3)

        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'Sample {i+1} - Channel 0')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate(config, args):
    """主评估函数"""
    print("=" * 80)
    print("IEclaude 评估脚本")
    print("=" * 80)

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 输出目录
    output_dir = config['train']['output_dir']
    print(f"输出目录: {output_dir}")

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

    # 创建测试数据加载器
    test_loader = create_dataloader(
        data=test_data,
        missing_rate=config['train']['missing_rate'],
        missing_pattern=config['train'].get('missing_pattern', 'random'),
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"\n测试集batch数量: {len(test_loader)}")

    # ========================================================================
    # 2. 加载模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 加载模型")
    print("=" * 80)

    model = IEDiffusionModel(
        in_channels=data_info['num_channels'],
        res_channels=config['model']['res_channels'],
        skip_channels=config['model']['skip_channels'],
        out_channels=data_info['num_channels'],
        num_res_layers=config['model']['num_res_layers'],
        dilation_cycle=config['model'].get('dilation_cycle', 10),
        diffusion_step_embed_dim_in=config['diffusion']['embed_dim_in'],
        diffusion_step_embed_dim_mid=config['diffusion']['embed_dim_mid'],
        diffusion_step_embed_dim_out=config['diffusion']['embed_dim_out'],
        tcn_channels=config['model']['tcn_channels'],
        tcn_kernel_size=config['model']['tcn_kernel_size'],
        tcn_dilation_rates=config['model']['tcn_dilation_rates'],
        tcn_dropout=config['model'].get('tcn_dropout', 0.0),
        s4_d_state=config['model']['s4_d_state'],
        s4_n_layers=config['model']['s4_n_layers'],
        s4_l_max=config['data']['seq_len'],
        s4_dropout=config['model'].get('s4_dropout', 0.0),
        s4_bidirectional=config['model'].get('s4_bidirectional', True)
    ).to(device)

    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.6f}")

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

    # ========================================================================
    # 4. 执行插补
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 执行插补")
    print("=" * 80)

    predictions, ground_truth, masks = impute(
        model=model,
        dataloader=test_loader,
        diffusion_process=diffusion_process,
        device=device,
        num_samples=args.num_samples,
        verbose=True
    )

    print(f"\n插补完成！")
    print(f"  预测形状: {predictions.shape}")

    # ========================================================================
    # 5. 计算指标
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. 计算指标")
    print("=" * 80)

    metrics = calculate_metrics(predictions, ground_truth, masks)

    print(f"\n评估指标 (缺失率={config['train']['missing_rate']*100:.0f}%):")
    print(f"  MAE:  {metrics['MAE']:.3f}")
    print(f"  RMSE: {metrics['RMSE']:.3f}")
    print(f"  缺失样本数: {metrics['num_missing']:,}")

    # 保存指标
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Missing Rate: {config['train']['missing_rate']*100:.0f}%\n")
        f.write(f"MAE: {metrics['MAE']:.3f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.3f}\n")
        f.write(f"Num Missing: {metrics['num_missing']}\n")

    print(f"\n指标已保存: {metrics_file}")

    # ========================================================================
    # 6. 可视化
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. 生成可视化")
    print("=" * 80)

    vis_path = os.path.join(output_dir, 'imputation_visualization.png')
    visualize_imputation(predictions, ground_truth, masks, vis_path, num_examples=5)
    print(f"可视化已保存: {vis_path}")

    # ========================================================================
    # 7. 完成
    # ========================================================================
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    evaluate(config, args)
