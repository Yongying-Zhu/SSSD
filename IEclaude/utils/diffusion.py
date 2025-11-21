"""
扩散过程工具类 (Diffusion Process Utils)

功能说明：
    实现DDPM (Denoising Diffusion Probabilistic Models) 的前向和反向扩散过程

核心概念：
    扩散模型通过逐步添加噪声将数据转换为纯噪声（前向过程），
    然后训练神经网络学习逐步去噪（反向过程），从而实现数据生成和插补

前向扩散过程（加噪）：
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    数学含义：
        - x_0: 原始干净数据
        - x_t: 第t步的噪声数据
        - alpha_bar_t: 累积噪声系数
        - 随着t增大，x_t越来越接近纯高斯噪声

反向扩散过程（去噪）：
    p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

    数学含义：
        - 神经网络预测噪声 epsilon_theta(x_t, t)
        - 使用预测的噪声计算 x_{t-1}
        - 逐步从噪声恢复到干净数据

超参数调节指南：
    1. 扩散步数 (T):
       - T越大，扩散过程越平滑，但计算量越大
       - 建议: 100-1000
       - 对于插补任务，200-500通常足够

    2. 噪声调度 (beta):
       - beta_0: 起始噪声水平 [建议: 0.0001-0.001]
       - beta_T: 结束噪声水平 [建议: 0.02-0.1]
       - 线性调度: beta_t = beta_0 + (beta_T - beta_0) * t / T
       - 余弦调度: 更平滑的噪声增长曲线

    3. 采样策略：
       - DDPM: 标准采样，需要T步
       - DDIM: 加速采样，可以跳过一些步骤
       - 对于插补，建议使用DDPM以获得更好的质量

调优建议：
    - 如果生成质量不好：增大T，调整beta范围
    - 如果训练不稳定：减小beta_T，增加warm-up
    - 如果采样太慢：使用DDIM或减小T
"""

import torch
import torch.nn as nn
import numpy as np


class DiffusionProcess:
    """
    扩散过程类

    功能：
        - 定义前向扩散过程（加噪）
        - 定义反向扩散过程（去噪）
        - 计算扩散相关的参数

    参数：
        T (int): 扩散步数
        beta_0 (float): 起始噪声水平
        beta_T (float): 结束噪声水平
        schedule (str): 噪声调度方式 ['linear', 'cosine']
    """
    def __init__(self, T=200, beta_0=0.0001, beta_T=0.02, schedule='linear'):
        self.T = T
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.schedule = schedule

        # 计算噪声调度
        self.betas = self._get_noise_schedule()

        # 计算相关参数
        self.alphas = 1.0 - self.betas  # alpha_t = 1 - beta_t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # alpha_bar_t = prod(alpha_i) for i=1..t

        # 计算其他有用的参数
        self.alpha_bars_prev = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])  # alpha_bar_{t-1}
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)  # sqrt(alpha_bar_t)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)  # sqrt(1 - alpha_bar_t)

        # 反向过程的参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )  # 后验方差

    def _get_noise_schedule(self):
        """
        计算噪声调度

        返回：
            betas (torch.Tensor): 噪声调度 [T]
        """
        if self.schedule == 'linear':
            # 线性调度：beta从beta_0线性增长到beta_T
            betas = torch.linspace(self.beta_0, self.beta_T, self.T)

        elif self.schedule == 'cosine':
            # 余弦调度：更平滑的增长曲线
            # 参考: Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)
            s = 0.008  # 小的偏移量
            steps = self.T + 1
            x = torch.linspace(0, self.T, steps)
            alphas_bar = torch.cos(((x / self.T) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)  # 裁剪到合理范围

        else:
            raise ValueError(f"未知的噪声调度方式: {self.schedule}")

        return betas

    def q_sample(self, x_0, t, noise=None):
        """
        前向扩散过程：从x_0采样x_t

        数学公式：
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            其中 epsilon ~ N(0, I)

        参数：
            x_0 (torch.Tensor): 原始数据 [B, C, L]
            t (torch.Tensor): 时间步 [B]
            noise (torch.Tensor, optional): 噪声 [B, C, L]

        返回：
            x_t (torch.Tensor): 加噪后的数据 [B, C, L]
            noise (torch.Tensor): 添加的噪声 [B, C, L]
        """
        # 如果未提供噪声，则采样标准高斯噪声
        if noise is None:
            noise = torch.randn_like(x_0)

        # 获取对应时间步的参数
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x_0.shape)

        # 计算 x_t
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def p_sample(self, model, x_t, t, observed_data, mask):
        """
        反向扩散过程：从x_t采样x_{t-1}

        数学公式：
            1. 使用模型预测噪声: epsilon_theta = model(x_t, observed_data, mask, t)
            2. 计算均值: mu = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta)
            3. 采样: x_{t-1} = mu + sigma_t * z, 其中 z ~ N(0, I)

        参数：
            model (nn.Module): 去噪模型
            x_t (torch.Tensor): 当前噪声数据 [B, C, L]
            t (torch.Tensor): 当前时间步 [B]
            observed_data (torch.Tensor): 观测数据 [B, C, L]
            mask (torch.Tensor): mask [B, C, L]

        返回：
            x_{t-1} (torch.Tensor): 去噪一步后的数据 [B, C, L]
        """
        # 使用模型预测噪声
        predicted_noise = model(x_t, observed_data, mask, t)

        # 获取对应时间步的参数
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        alpha_bar_t = self._extract(self.alpha_bars, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)

        # 计算均值
        # mu_t = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x_t - coef2 * predicted_noise)

        # 计算方差
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)

        # 采样 x_{t-1}
        if t[0] > 0:
            # 如果不是最后一步，添加噪声
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(posterior_var) * noise
        else:
            # 如果是最后一步 (t=0)，不添加噪声
            x_t_minus_1 = mean

        return x_t_minus_1

    def p_sample_loop(self, model, shape, observed_data, mask, device='cpu', verbose=False):
        """
        完整的反向扩散循环：从纯噪声逐步生成数据

        参数：
            model (nn.Module): 去噪模型
            shape (tuple): 数据形状 (B, C, L)
            observed_data (torch.Tensor): 观测数据 [B, C, L]
            mask (torch.Tensor): mask [B, C, L]
            device (str): 设备
            verbose (bool): 是否显示进度

        返回：
            x_0 (torch.Tensor): 生成的数据 [B, C, L]
        """
        B, C, L = shape

        # 从纯噪声开始
        x_t = torch.randn(B, C, L, device=device)

        # 逐步去噪
        for t_idx in reversed(range(self.T)):
            # 当前时间步
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # 去噪一步
            x_t = self.p_sample(model, x_t, t, observed_data, mask)

            # 打印进度
            if verbose and (t_idx % 20 == 0 or t_idx == 0):
                print(f"  去噪进度: {self.T - t_idx}/{self.T}")

        return x_t

    def _extract(self, a, t, x_shape):
        """
        从数组a中提取对应时间步t的值，并reshape到与x相同的形状

        参数：
            a (torch.Tensor): 参数数组 [T]
            t (torch.Tensor): 时间步 [B]
            x_shape (tuple): 目标形状

        返回：
            out (torch.Tensor): 提取并reshape后的值
        """
        batch_size = t.shape[0]
        out = a.to(t.device)[t]  # [B]
        # Reshape到 [B, 1, 1, ...] 以便广播
        return out.view(batch_size, *([1] * (len(x_shape) - 1)))


class DiffusionLoss(nn.Module):
    """
    扩散模型损失函数

    功能：
        计算预测噪声和真实噪声之间的均方误差损失

    损失公式：
        L = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]

    其中：
        - epsilon: 真实添加的噪声
        - epsilon_theta: 模型预测的噪声
        - x_t: 通过前向扩散得到的噪声数据

    参数：
        only_generate_missing (bool): 是否只对缺失位置计算损失
            - True: 只计算缺失位置的损失（适用于插补任务）
            - False: 计算所有位置的损失（适用于生成任务）
    """
    def __init__(self, only_generate_missing=True):
        super().__init__()
        self.only_generate_missing = only_generate_missing

    def forward(self, predicted_noise, true_noise, mask=None):
        """
        计算损失

        参数：
            predicted_noise (torch.Tensor): 模型预测的噪声 [B, C, L]
            true_noise (torch.Tensor): 真实噪声 [B, C, L]
            mask (torch.Tensor, optional): mask [B, C, L]
                                          1=观测，0=缺失

        返回：
            loss (torch.Tensor): 损失值（标量）
        """
        if self.only_generate_missing and mask is not None:
            # 只计算缺失位置的损失
            # mask中1表示观测，0表示缺失
            # 我们需要计算缺失位置，所以用 (1 - mask)
            missing_mask = (1 - mask).float()

            # 计算缺失位置的MSE
            loss = ((predicted_noise - true_noise) ** 2 * missing_mask).sum()

            # 归一化：除以缺失位置的数量
            num_missing = missing_mask.sum()
            if num_missing > 0:
                loss = loss / num_missing
        else:
            # 计算所有位置的MSE
            loss = ((predicted_noise - true_noise) ** 2).mean()

        return loss


# ============================================================================
# 使用示例和调试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试扩散过程
    """
    print("=" * 80)
    print("扩散过程测试")
    print("=" * 80)

    # 创建扩散过程
    diffusion = DiffusionProcess(T=200, beta_0=0.0001, beta_T=0.02, schedule='linear')

    print(f"\n扩散过程配置:")
    print(f"  扩散步数 T: {diffusion.T}")
    print(f"  起始噪声 beta_0: {diffusion.beta_0}")
    print(f"  结束噪声 beta_T: {diffusion.beta_T}")
    print(f"  调度方式: {diffusion.schedule}")

    # 创建模拟数据
    batch_size = 4
    channels = 370
    seq_len = 168

    x_0 = torch.randn(batch_size, channels, seq_len)
    print(f"\n原始数据 x_0: {x_0.shape}")

    # 测试前向扩散
    print(f"\n测试前向扩散...")
    t = torch.randint(0, diffusion.T, (batch_size,))
    x_t, noise = diffusion.q_sample(x_0, t)
    print(f"  时间步 t: {t}")
    print(f"  加噪数据 x_t: {x_t.shape}")
    print(f"  噪声: {noise.shape}")

    # 可视化不同时间步的噪声水平
    print(f"\n不同时间步的噪声系数:")
    test_steps = [0, 50, 100, 150, 199]
    for step in test_steps:
        sqrt_alpha_bar = diffusion.sqrt_alpha_bars[step]
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alpha_bars[step]
        print(f"  t={step:3d}: sqrt(alpha_bar)={sqrt_alpha_bar:.4f}, sqrt(1-alpha_bar)={sqrt_one_minus_alpha_bar:.4f}")

    # 测试损失函数
    print(f"\n测试损失函数...")
    loss_fn = DiffusionLoss(only_generate_missing=True)

    # 创建mask
    mask = torch.rand(batch_size, channels, seq_len) > 0.3
    predicted_noise = torch.randn_like(noise)

    loss = loss_fn(predicted_noise, noise, mask)
    print(f"  损失值: {loss.item():.6f}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
