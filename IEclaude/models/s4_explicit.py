"""
S4显式特征提取模块 (Structured State Space Model for Explicit Feature Extraction)

功能说明：
    使用结构化状态空间模型提取时间序列的长期依赖特征

核心概念：
    S4是一种序列建模方法，通过状态空间模型捕获长期依赖关系

    状态空间模型的离散形式：
        x_{t+1} = A * x_t + B * u_t   (状态更新方程)
        y_t = C * x_t + D * u_t        (输出方程)

    其中：
        - u_t: 输入序列 [input sequence]
        - x_t: 隐藏状态 [hidden state]
        - y_t: 输出序列 [output sequence]
        - A: 状态转移矩阵 [NxN] - 控制历史状态如何演化
        - B: 输入矩阵 [Nx1] - 控制当前输入如何影响状态
        - C: 输出矩阵 [1xN] - 控制状态如何映射到输出
        - D: 前馈矩阵 [标量] - 控制输入到输出的直接连接（skip connection）

超参数调节指南：
    1. 基础超参数：
       - d_model: 特征维度，即隐藏层大小 [建议: 64-512]
       - d_state (N): 状态维度，控制模型容量 [建议: 16-128]
         * 增大N可以提升模型对复杂序列的建模能力
         * 但会增加计算量，建议从64开始
       - dropout: 防止过拟合 [建议: 0.0-0.2]

    2. 架构调整 - ABCD矩阵：
       - A矩阵初始化方式：
         * 'hippo': HiPPO初始化，适合长期依赖（推荐）
         * 'diagonal': 对角化初始化，计算更快但表达能力较弱
         * 'random': 随机初始化

       - B, C矩阵：
         * 一般使用标准正态分布初始化
         * 可以通过学习率控制这些参数的学习速度

       - D矩阵：
         * 相当于残差连接的权重
         * 通常初始化为较小的值（0.1-1.0）

    3. 序列长度：
       - l_max: 最大序列长度 [建议: 设置为实际序列长度]
         * 对于交通数据：如果用168小时（1周），设为168
         * 如果不确定，可以设为2的幂次（如256, 512）

    4. 双向模式：
       - bidirectional: 是否使用双向S4 [True/False]
         * True: 同时考虑过去和未来的信息（适合插补任务）
         * False: 只考虑过去的信息（适合预测任务）

调优建议：
    - 如果训练不稳定：减小学习率，增大d_state
    - 如果效果不好：尝试双向模式，增大d_state
    - 如果过拟合：增大dropout，减小d_state
    - 如果计算太慢：减小d_state，使用单向模式
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special as ss
import numpy as np


def hippo_initializer(N):
    """
    HiPPO (High-order Polynomial Projection Operators) 初始化

    功能：
        生成适合建模长期依赖的A和B矩阵初始值

    原理：
        HiPPO理论证明，通过特定的A、B矩阵设计，可以使状态向量
        高效地记忆和压缩历史信息，实现对长序列的有效建模

    参数：
        N (int): 状态维度

    返回：
        A (np.array): 状态转移矩阵 [N, N]
        B (np.array): 输入矩阵 [N, 1]

    数学推导：
        HiPPO矩阵的构造基于Legendre多项式，能够将历史信息
        投影到正交多项式基上，实现信息的有效压缩和记忆
    """
    # HiPPO-LegS 矩阵构造
    # 这是HiPPO论文中推荐的矩阵形式，特别适合长期依赖建模

    # 创建NxN的下三角矩阵
    # A矩阵的(n,k)元素定义如下：
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = n + 1

    # B矩阵定义为每个元素 B[n] = sqrt(2*n + 1)
    B = np.sqrt(2 * np.arange(N) + 1).reshape(N, 1)

    return A, B


def discretize_zoh(A, B, dt):
    """
    零阶保持器离散化 (Zero-Order Hold Discretization)

    功能：
        将连续状态空间模型离散化为离散状态空间模型

    原理：
        连续系统: dx/dt = A*x + B*u
        离散系统: x_{k+1} = A_bar*x_k + B_bar*u_k

        零阶保持假设输入信号在采样间隔内保持恒定

    参数：
        A (torch.Tensor): 连续状态转移矩阵 [N, N]
        B (torch.Tensor): 连续输入矩阵 [N, 1]
        dt (float): 采样时间步长（离散化步长）

    返回：
        A_bar (torch.Tensor): 离散状态转移矩阵 [N, N]
        B_bar (torch.Tensor): 离散输入矩阵 [N, 1]

    数学公式：
        A_bar = exp(A * dt)
        B_bar = (A^{-1})(exp(A*dt) - I) * B

    调节建议：
        - dt越小，离散化越精确，但可能需要更多时间步
        - dt越大，计算越快，但可能损失精度
        - 一般设置dt=1.0即可
    """
    # 计算矩阵指数 exp(A * dt)
    # 这里使用特征值分解来高效计算
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    A_bar = torch.matrix_exp(A * dt)

    # 计算 B_bar = (A^{-1})(exp(A*dt) - I) * B
    # 如果A可逆，使用该公式；否则使用近似
    try:
        A_inv = torch.linalg.inv(A)
        B_bar = A_inv @ (A_bar - I) @ B
    except:
        # 如果A不可逆，使用一阶近似：B_bar ≈ dt * B
        B_bar = dt * B

    return A_bar, B_bar


class S4Kernel(nn.Module):
    """
    S4卷积核模块 (S4 Convolutional Kernel)

    功能：
        生成S4的卷积核，用于高效处理长序列

    原理：
        S4通过将状态空间模型转换为卷积形式，可以利用FFT加速计算
        卷积核K的每个元素: K_l = C * A^l * B
        这样，输出可以表示为: y = K * u (卷积操作)

    参数：
        d_model (int): 特征维度
        d_state (int): 状态维度 N
        l_max (int): 最大序列长度
        dt_min (float): 最小时间步长
        dt_max (float): 最大时间步长

    ABCD矩阵调节：
        - 通过修改A的初始化方式改变记忆模式
        - 通过修改B、C的初始化改变输入输出的映射
        - 通过log_dt调节时间尺度
    """
    def __init__(self, d_model, d_state=64, l_max=1, dt_min=0.001, dt_max=0.1):
        super().__init__()

        self.d_model = d_model  # H：特征维度
        self.d_state = d_state  # N：状态维度
        self.l_max = l_max      # L：最大序列长度

        # === A矩阵：状态转移矩阵 [N, N] ===
        # 使用HiPPO初始化，适合长期依赖
        A, B_init = hippo_initializer(d_state)
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))

        # === B矩阵：输入矩阵 [N, 1] ===
        # 控制当前输入如何影响状态
        # 使用HiPPO推荐的初始化
        self.B = nn.Parameter(torch.tensor(B_init, dtype=torch.float32))

        # === C矩阵：输出矩阵 [1, N] ===
        # 控制状态如何映射到输出
        # 使用标准正态分布初始化
        C_init = torch.randn(1, d_state) / np.sqrt(d_state)
        self.C = nn.Parameter(C_init)

        # === log_dt: 对数时间步长 ===
        # 通过学习时间步长，模型可以自适应调整时间尺度
        # dt越大，状态更新越激进；dt越小，状态更新越保守
        log_dt = torch.rand(d_model) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # === D矩阵：前馈矩阵（skip connection） ===
        # 允许输入直接影响输出，类似于残差连接
        # 初始化为1.0，可以调节以控制skip connection的强度
        self.D = nn.Parameter(torch.ones(d_model))

    def kernel(self, L):
        """
        生成长度为L的S4卷积核

        参数：
            L (int): 目标序列长度

        返回：
            K (torch.Tensor): 卷积核 [d_model, L]

        计算流程：
            1. 离散化A、B矩阵
            2. 计算K_l = C * A^l * B for l=0,1,...,L-1
            3. 返回卷积核
        """
        # 计算时间步长 dt = exp(log_dt)
        dt = torch.exp(self.log_dt)  # [d_model]

        # 为每个特征维度离散化状态空间
        # 这里简化处理：使用平均dt进行离散化
        dt_mean = dt.mean()

        # 离散化A和B矩阵
        A_bar, B_bar = discretize_zoh(self.A, self.B, dt_mean)

        # 计算卷积核 K_l = C * (A_bar)^l * B_bar
        # 使用幂迭代方法计算
        K = []
        A_power = torch.eye(self.d_state, device=self.A.device, dtype=self.A.dtype)

        for l in range(L):
            # K_l = C * A^l * B
            K_l = self.C @ A_power @ B_bar  # [1, N] @ [N, N] @ [N, 1] = [1, 1]
            K.append(K_l.squeeze())

            # 更新 A^l 为 A^{l+1}
            A_power = A_power @ A_bar

        # 将列表转换为张量 [L]
        K = torch.stack(K, dim=0)  # [L]

        # 扩展到所有特征维度 [d_model, L]
        K = K.unsqueeze(0).expand(self.d_model, -1)

        return K

    def forward(self, L):
        """
        前向传播：生成卷积核

        参数：
            L (int): 序列长度

        返回：
            K (torch.Tensor): 卷积核 [d_model, L]
        """
        return self.kernel(L)


class S4Layer(nn.Module):
    """
    S4层 (S4 Layer)

    功能：
        完整的S4层，包括S4卷积、激活函数、dropout和输出线性变换

    结构：
        输入 -> S4卷积 -> 激活 -> Dropout -> 线性变换 -> 输出

    参数：
        d_model (int): 特征维度
        d_state (int): 状态维度N
        l_max (int): 最大序列长度
        dropout (float): Dropout比率
        bidirectional (bool): 是否使用双向S4
        activation (str): 激活函数类型 ['gelu', 'relu', 'swish']

    使用建议：
        - 对于插补任务，建议使用bidirectional=True
        - 对于预测任务，建议使用bidirectional=False
        - 激活函数通常选择'gelu'，效果较好
    """
    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=1,
        dropout=0.0,
        bidirectional=True,
        activation='gelu'
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional

        # S4卷积核
        self.kernel = S4Kernel(d_model, d_state, l_max)

        # 如果使用双向，需要两个kernel（前向和后向）
        if bidirectional:
            self.kernel_backward = S4Kernel(d_model, d_state, l_max)

        # 激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # Swish = SiLU
        else:
            self.activation = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 输出线性变换
        # 如果是双向，特征维度会翻倍
        output_dim = d_model * 2 if bidirectional else d_model
        self.output_linear = nn.Linear(output_dim, d_model)

        # Layer Normalization：稳定训练
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        """
        前向传播

        参数：
            u (torch.Tensor): 输入 [batch_size, d_model, seq_len]

        返回：
            y (torch.Tensor): 输出 [batch_size, d_model, seq_len]

        计算流程：
            1. 生成S4卷积核
            2. 使用FFT进行高效卷积
            3. 添加skip connection (D矩阵)
            4. 激活、dropout和输出变换
        """
        B, H, L = u.shape
        assert H == self.d_model

        # === 生成卷积核 ===
        k = self.kernel(L)  # [d_model, L]

        # === 前向卷积 ===
        # 使用FFT加速卷积运算
        # 卷积定理: conv(x, k) = IFFT(FFT(x) * FFT(k))
        k_f = torch.fft.rfft(k, n=2*L)  # [d_model, L_fft]
        u_f = torch.fft.rfft(u, n=2*L, dim=-1)  # [B, d_model, L_fft]
        y_f = u_f * k_f.unsqueeze(0)  # [B, d_model, L_fft]
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]  # [B, d_model, L]

        # === 双向处理 ===
        if self.bidirectional:
            # 后向卷积：反转序列
            k_b = self.kernel_backward(L)
            k_b_f = torch.fft.rfft(k_b, n=2*L)
            u_reversed = torch.flip(u, dims=[-1])  # 反转序列
            u_reversed_f = torch.fft.rfft(u_reversed, n=2*L, dim=-1)
            y_b_f = u_reversed_f * k_b_f.unsqueeze(0)
            y_b = torch.fft.irfft(y_b_f, n=2*L)[..., :L]
            y_b = torch.flip(y_b, dims=[-1])  # 反转回来

            # 拼接前向和后向
            y = torch.cat([y, y_b], dim=1)  # [B, 2*d_model, L]

        # === 添加skip connection (D矩阵) ===
        D = self.kernel.D.unsqueeze(0).unsqueeze(-1)  # [1, d_model, 1]
        y = y + u * D  # [B, d_model, L] 或 [B, 2*d_model, L]

        # === 激活和Dropout ===
        y = self.activation(y)
        y = self.dropout(y)

        # === 输出线性变换 ===
        # 转换为 [B, L, d_model] 格式进行线性变换
        y = y.transpose(1, 2)  # [B, L, d_model] 或 [B, L, 2*d_model]
        y = self.output_linear(y)  # [B, L, d_model]

        # === Layer Normalization + Residual Connection ===
        # 转换回 [B, d_model, L]
        y = y.transpose(1, 2)  # [B, d_model, L]

        # 残差连接
        u_norm = self.norm(u.transpose(1, 2)).transpose(1, 2)
        y = y + u_norm

        return y


class S4ExplicitExtractor(nn.Module):
    """
    S4显式特征提取器 (S4 Explicit Feature Extractor)

    功能：
        堆叠多个S4层，提取长期依赖的显式特征

    架构：
        输入 -> [S4Layer1, S4Layer2, ..., S4LayerN] -> 输出

    参数：
        d_model (int): 特征维度
        d_state (int): 状态维度N
        n_layers (int): S4层数
        l_max (int): 最大序列长度
        dropout (float): Dropout比率
        bidirectional (bool): 是否使用双向S4

    调优建议：
        1. 增加n_layers可以提取更抽象的特征（2-6层）
        2. 增大d_state可以提升建模能力（32-128）
        3. 对于长序列（>100），建议使用bidirectional=True
        4. 如果过拟合，增大dropout（0.1-0.2）
    """
    def __init__(
        self,
        d_model,
        d_state=64,
        n_layers=4,
        l_max=1,
        dropout=0.0,
        bidirectional=True
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers

        # 堆叠多个S4层
        self.layers = nn.ModuleList([
            S4Layer(
                d_model=d_model,
                d_state=d_state,
                l_max=l_max,
                dropout=dropout,
                bidirectional=bidirectional
            )
            for _ in range(n_layers)
        ])

        # 最终的Layer Normalization
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        前向传播

        参数：
            x (torch.Tensor): 输入 [batch_size, d_model, seq_len]

        返回：
            explicit_features (torch.Tensor): 显式特征 [batch_size, d_model, seq_len]
        """
        # 依次通过所有S4层
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        # 转换为 [B, L, d_model] 进行LayerNorm
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        x = x.transpose(1, 2)  # 转换回 [B, d_model, L]

        return x

    def get_config(self):
        """
        获取模型配置信息

        返回：
            config (dict): 包含模型配置的字典
        """
        return {
            'd_model': self.d_model,
            'd_state': self.d_state,
            'n_layers': self.n_layers,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# 使用示例和调试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试S4显式特征提取器
    """
    print("=" * 80)
    print("S4显式特征提取器测试")
    print("=" * 80)

    # 模拟输入数据
    batch_size = 4
    d_model = 256      # 特征维度
    seq_len = 168      # 序列长度（1周）

    # 创建随机输入
    x = torch.randn(batch_size, d_model, seq_len)
    print(f"\n输入张量形状: {x.shape}")

    # 创建S4模型
    model = S4ExplicitExtractor(
        d_model=d_model,
        d_state=64,        # 状态维度
        n_layers=4,        # 4层S4
        l_max=seq_len,     # 最大序列长度
        dropout=0.1,
        bidirectional=True # 双向
    )

    # 前向传播
    output = model(x)
    print(f"输出张量形状: {output.shape}")

    # 显示配置
    config = model.get_config()
    print(f"\n模型配置:")
    print(f"  特征维度: {config['d_model']}")
    print(f"  状态维度: {config['d_state']}")
    print(f"  层数: {config['n_layers']}")
    print(f"  参数总数: {config['num_parameters']:,}")
    print(f"  可训练参数: {config['num_trainable_parameters']:,}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
