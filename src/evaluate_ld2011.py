"""
隐式-显式扩散模型评估脚本
用于LD2011_2014数据集的时间序列插补评估

该脚本支持：
1. 计算MAE和RMSE指标
2. 不同缺失率的评估（20%-80%）
3. 可视化插补结果
4. 生成评估报告表格

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
import pandas as pd

# 导入工具函数
from utils.util import (
    find_max_epoch,
    sampling,
    calc_diffusion_hyperparams
)

# 导入模型
from imputers.ImplicitExplicitDiffusion import ImplicitExplicitDiffusion


def create_random_mask(data, missing_ratio):
    """
    创建随机缺失掩码

    参数:
        data: 输入数据，形状 [batch, channels, length]
        missing_ratio: 缺失率

    返回:
        mask: 掩码张量
    """
    mask = torch.ones_like(data)
    batch_size, channels, length = data.shape

    for b in range(batch_size):
        for c in range(channels):
            n_missing = int(length * missing_ratio)
            missing_indices = torch.randperm(length)[:n_missing]
            mask[b, c, missing_indices] = 0

    return mask


def calculate_metrics(predictions, ground_truth, mask):
    """
    计算评估指标

    参数:
        predictions: 预测值，形状 [batch, channels, length]
        ground_truth: 真实值，形状 [batch, channels, length]
        mask: 掩码，形状 [batch, channels, length]
              0表示缺失位置（需要评估的位置）

    返回:
        mae: 平均绝对误差
        rmse: 均方根误差
    """
    # 只在缺失位置计算误差
    missing_mask = (mask == 0)

    # 提取缺失位置的预测值和真实值
    pred_missing = predictions[missing_mask]
    true_missing = ground_truth[missing_mask]

    # 计算MAE
    mae = torch.mean(torch.abs(pred_missing - true_missing)).item()

    # 计算RMSE
    mse = torch.mean((pred_missing - true_missing) ** 2).item()
    rmse = np.sqrt(mse)

    return mae, rmse


def evaluate(ckpt_path,
             missing_ratio,
             n_samples,
             device,
             save_visualizations=True):
    """
    评估模型性能

    参数:
        ckpt_path: 检查点路径
        missing_ratio: 缺失率
        n_samples: 评估的样本数量
        device: 计算设备
        save_visualizations: 是否保存可视化结果

    返回:
        mae: 平均绝对误差
        rmse: 均方根误差
    """
    # ===== 1. 设置设备 =====
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 将扩散超参数移到设备
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # ===== 2. 创建模型 =====
    print("\n创建模型...")
    net = ImplicitExplicitDiffusion(**model_config).to(device)

    # ===== 3. 加载检查点 =====
    # 查找最新的检查点
    ckpt_iter = find_max_epoch(ckpt_path)

    if ckpt_iter < 0:
        print("错误: 未找到有效的检查点")
        return None, None

    try:
        model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f'成功加载检查点: 迭代 {ckpt_iter}')
    except Exception as e:
        print(f'加载检查点失败: {e}')
        return None, None

    # 设置为评估模式
    net.eval()

    # ===== 4. 加载测试数据 =====
    print("\n加载测试数据...")
    test_data = np.load(testset_config['test_data_path'])
    test_data = torch.from_numpy(test_data).float()
    test_data = test_data.permute(0, 2, 1)  # [N, C, L]

    # 限制评估样本数
    if n_samples > 0 and n_samples < len(test_data):
        test_data = test_data[:n_samples]

    print(f"测试数据形状: {test_data.shape}")

    # ===== 5. 执行评估 =====
    print(f"\n开始评估 (缺失率: {missing_ratio * 100}%)...")

    all_mae = []
    all_rmse = []

    # 用于可视化的样本
    vis_samples = min(5, len(test_data))
    vis_predictions = []
    vis_ground_truth = []
    vis_masks = []

    with torch.no_grad():
        # 批量处理
        batch_size = 16
        n_batches = (len(test_data) + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="评估进度"):
            # 获取批次数据
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(test_data))
            batch = test_data[start_idx:end_idx].to(device)

            # 创建缺失掩码
            mask = create_random_mask(batch, missing_ratio)
            mask = mask.to(device)

            # 创建观测数据（缺失位置置零）
            observed_data = batch * mask

            # 执行采样（扩散模型的逆向过程）
            predictions = sampling(
                net,
                batch.size(),
                diffusion_hyperparams,
                cond=observed_data,
                mask=mask,
                only_generate_missing=1
            )

            # 计算指标
            mae, rmse = calculate_metrics(predictions, batch, mask)
            all_mae.append(mae)
            all_rmse.append(rmse)

            # 保存前几个样本用于可视化
            if i == 0 and save_visualizations:
                n_vis = min(vis_samples, len(batch))
                vis_predictions.append(predictions[:n_vis].cpu())
                vis_ground_truth.append(batch[:n_vis].cpu())
                vis_masks.append(mask[:n_vis].cpu())

    # ===== 6. 计算平均指标 =====
    avg_mae = np.mean(all_mae)
    avg_rmse = np.mean(all_rmse)

    print("\n" + "=" * 60)
    print(f"评估结果 (缺失率: {missing_ratio * 100}%)")
    print("=" * 60)
    print(f"MAE:  {avg_mae:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    print("=" * 60)

    # ===== 7. 可视化结果 =====
    if save_visualizations and len(vis_predictions) > 0:
        print("\n保存可视化结果...")

        # 创建输出目录
        vis_dir = os.path.join(ckpt_path, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        vis_predictions = torch.cat(vis_predictions, dim=0)
        vis_ground_truth = torch.cat(vis_ground_truth, dim=0)
        vis_masks = torch.cat(vis_masks, dim=0)

        # 为每个样本创建可视化
        for idx in range(min(vis_samples, len(vis_predictions))):
            pred = vis_predictions[idx].numpy()  # [channels, length]
            true = vis_ground_truth[idx].numpy()
            mask = vis_masks[idx].numpy()

            # 选择前3个通道进行可视化
            n_channels_to_plot = min(3, pred.shape[0])

            fig, axes = plt.subplots(n_channels_to_plot, 1,
                                    figsize=(15, 3 * n_channels_to_plot))

            if n_channels_to_plot == 1:
                axes = [axes]

            for ch in range(n_channels_to_plot):
                ax = axes[ch]

                # 时间轴
                time_steps = np.arange(pred.shape[1])

                # 绘制真实值
                ax.plot(time_steps, true[ch], 'b-', linewidth=2,
                       label='真实值', alpha=0.7)

                # 绘制预测值（只在缺失位置）
                missing_idx = np.where(mask[ch] == 0)[0]
                ax.scatter(missing_idx, pred[ch, missing_idx],
                          c='red', s=30, label='预测值（缺失位置）',
                          alpha=0.8, marker='o')

                # 绘制观测值
                observed_idx = np.where(mask[ch] == 1)[0]
                ax.scatter(observed_idx, true[ch, observed_idx],
                          c='green', s=10, label='观测值',
                          alpha=0.5, marker='x')

                ax.set_xlabel('时间步', fontsize=10)
                ax.set_ylabel(f'通道 {ch+1}', fontsize=10)
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.suptitle(
                f'样本 {idx+1} - 插补结果 (缺失率: {missing_ratio*100}%)\n'
                f'MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}',
                fontsize=12, fontweight='bold'
            )
            plt.tight_layout()

            # 保存图像
            save_path = os.path.join(
                vis_dir,
                f'sample_{idx+1}_missing_{int(missing_ratio*100)}.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"可视化结果已保存到: {vis_dir}")

    return avg_mae, avg_rmse


def evaluate_multiple_missing_ratios(ckpt_base_path,
                                     missing_ratios,
                                     n_samples,
                                     device):
    """
    评估多个缺失率下的模型性能

    参数:
        ckpt_base_path: 检查点基础路径
        missing_ratios: 缺失率列表，例如 [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        n_samples: 每个缺失率评估的样本数
        device: 计算设备

    返回:
        results_df: 包含所有结果的DataFrame
    """
    results = []

    for ratio in missing_ratios:
        print("\n" + "=" * 80)
        print(f"评估缺失率: {ratio * 100}%")
        print("=" * 80)

        # 构建检查点路径
        local_path = f"T{diffusion_config['T']}_beta0{diffusion_config['beta_0']}_betaT{diffusion_config['beta_T']}_missing{int(ratio * 100)}"
        ckpt_path = os.path.join(ckpt_base_path, local_path)

        if not os.path.exists(ckpt_path):
            print(f"警告: 检查点路径不存在: {ckpt_path}")
            print("跳过此缺失率...")
            continue

        # 执行评估
        mae, rmse = evaluate(
            ckpt_path=ckpt_path,
            missing_ratio=ratio,
            n_samples=n_samples,
            device=device,
            save_visualizations=True
        )

        if mae is not None and rmse is not None:
            results.append({
                'Missing Ratio': f'{int(ratio * 100)}%',
                'MAE': mae,
                'RMSE': rmse
            })

    # 创建DataFrame
    results_df = pd.DataFrame(results)

    # 保存结果表格
    if len(results) > 0:
        table_path = os.path.join(ckpt_base_path, 'evaluation_results.csv')
        results_df.to_csv(table_path, index=False)
        print(f"\n评估结果已保存到: {table_path}")

        # 打印表格
        print("\n" + "=" * 80)
        print("评估结果汇总")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)

        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # MAE图
        ax1.plot([int(r.split('%')[0]) for r in results_df['Missing Ratio']],
                results_df['MAE'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('缺失率 (%)', fontsize=12)
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_title('MAE vs 缺失率', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # RMSE图
        ax2.plot([int(r.split('%')[0]) for r in results_df['Missing Ratio']],
                results_df['RMSE'], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('缺失率 (%)', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('RMSE vs 缺失率', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(ckpt_base_path, 'metrics_vs_missing_ratio.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n指标对比图已保存到: {plot_path}")
        plt.close()

    return results_df


if __name__ == "__main__":
    """
    主函数

    使用方法:
    # 评估单个缺失率
    python evaluate_ld2011.py -c config/config_ImplicitExplicit_LD2011.json --missing_ratio 0.2

    # 评估多个缺失率
    python evaluate_ld2011.py -c config/config_ImplicitExplicit_LD2011.json --eval_all
    """
    parser = argparse.ArgumentParser(description='评估隐式-显式扩散模型')

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/config_ImplicitExplicit_LD2011.json',
        help='配置文件路径'
    )

    parser.add_argument(
        '--missing_ratio',
        type=float,
        default=0.2,
        help='缺失率 (0.0-1.0)'
    )

    parser.add_argument(
        '--eval_all',
        action='store_true',
        help='评估所有缺失率 (20%%-80%%)'
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=-1,
        help='评估的样本数量，-1表示使用全部测试集'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='计算设备'
    )

    args = parser.parse_args()

    # ===== 加载配置 =====
    with open(args.config) as f:
        config = json.loads(f.read())

    print("=" * 80)
    print("隐式-显式扩散模型 - 评估脚本")
    print("=" * 80)

    # 提取配置
    train_config = config["train_config"]
    global testset_config
    testset_config = config["trainset_config"]  # 测试集配置
    global diffusion_config
    diffusion_config = config["diffusion_config"]
    global model_config
    model_config = config["model_config"]

    # 计算扩散超参数
    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    # 获取检查点路径
    ckpt_base_path = train_config['output_directory']

    # 执行评估
    if args.eval_all:
        # 评估所有缺失率
        missing_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        print(f"\n评估所有缺失率: {[f'{int(r*100)}%' for r in missing_ratios]}")

        results_df = evaluate_multiple_missing_ratios(
            ckpt_base_path=ckpt_base_path,
            missing_ratios=missing_ratios,
            n_samples=args.n_samples,
            device=args.device
        )
    else:
        # 评估单个缺失率
        ratio = args.missing_ratio
        local_path = f"T{diffusion_config['T']}_beta0{diffusion_config['beta_0']}_betaT{diffusion_config['beta_T']}_missing{int(ratio * 100)}"
        ckpt_path = os.path.join(ckpt_base_path, local_path)

        mae, rmse = evaluate(
            ckpt_path=ckpt_path,
            missing_ratio=ratio,
            n_samples=args.n_samples,
            device=args.device,
            save_visualizations=True
        )

    print("\n评估完成！")
