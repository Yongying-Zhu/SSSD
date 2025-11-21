"""
隐式显式扩散模型 (Implicit-Explicit Diffusion Model for Traffic Data Imputation)

论文参考：
    A Diffusion Model for Traffic Data Imputation

功能说明：
    结合TCN隐式特征提取和S4显式特征提取的扩散模型，用于时间序列插补

核心创新：
    1. 隐式特征提取（TCN）：使用扩张因果卷积捕获多时间尺度的隐式依赖
    2. 显式特征提取（S4）：使用结构化状态空间模型捕获长期显式依赖
    3. 特征融合：将隐式和显式特征有机结合指导扩散去噪过程

模型架构：
    输入 -> DETACH分解 -> [隐式提取TCN, 显式提取S4] -> 特征融合
         -> 残差卷积层 + 扩散嵌入 -> 输出

超参数调节指南：
    1. 基础超参数：
       - in_channels: 输入通道数（传感器/特征数量）
       - res_channels: 残差层通道数 [建议: 128-512]
       - skip_channels: 跳跃连接通道数 [建议: 128-512]
       - num_res_layers: 残差层数量 [建议: 20-40]
       - dropout: Dropout比率 [建议: 0.0-0.2]

    2. 隐式模块调节（TCN）：
       - tcn_channels: TCN通道数列表 [256, 256, 256]
       - tcn_kernel_size: 卷积核大小 [3]
       - tcn_dilation_rates: 扩张率 [1, 2, 4, 8]
         * 调整扩张率可以改变时间尺度的捕获
         * 增加扩张率：[1, 2, 4, 8, 16] 捕获更长依赖
         * 跳跃增长：[1, 4, 16] 捕获稀疏长期依赖

    3. 显式模块调节（S4）：
       - s4_d_state: S4状态维度 [建议: 32-128]
       - s4_n_layers: S4层数 [建议: 2-6]
       - s4_bidirectional: 双向模式 [True for imputation]
       - S4的ABCD矩阵通过d_state和初始化方式控制

    4. 扩散模型参数：
       - diffusion_step_embed_dim: 扩散步骤嵌入维度
         * 用于编码当前处于扩散过程的哪一步
         * 增大可以提供更精细的步骤信息 [建议: 128-512]

    5. 训练技巧：
       - 如果训练不稳定：减小学习率，增加warm-up步数
       - 如果效果不好：增大模型容量（res_channels, num_res_layers）
       - 如果过拟合：增大dropout，使用数据增强
       - 如果计算太慢：减小num_res_layers，使用单向S4
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入TCN和S4模块
from .tcn_implicit import TCNImplicitExtractor
from .s4_explicit import S4ExplicitExtractor


def swish(x):
    """
    Swish激活函数: f(x) = x * sigmoid(x)

    特点：
        - 平滑的非线性函数
        - 自门控（self-gating）机制
        - 通常比ReLU效果更好
    """
    return x * torch.sigmoid(x)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    计算扩散步骤的位置编码 (Diffusion Step Embedding)

    功能：
        将扩散步骤数（整数）编码为高维向量，类似于Transformer的位置编码

    原理：
        使用正弦和余弦函数的组合，为每个扩散步骤生成唯一的表示
        这使模型能够识别当前处于扩散过程的哪个阶段

    参数：
        diffusion_steps (torch.Tensor): 扩散步骤 [batch_size] 或 [batch_size, 1]
        diffusion_step_embed_dim_in (int): 嵌入维度

    返回：
        embed (torch.Tensor): 扩散步骤嵌入 [batch_size, diffusion_step_embed_dim_in]

    数学公式：
        PE(step, 2i) = sin(step / 10000^(2i/d))
        PE(step, 2i+1) = cos(step / 10000^(2i/d))
    """
    assert diffusion_step_embed_dim_in % 2 == 0, "嵌入维度必须是偶数"

    # 确保diffusion_steps是1维张量
    if len(diffusion_steps.shape) == 0:
        diffusion_steps = diffusion_steps.unsqueeze(0)
    elif len(diffusion_steps.shape) == 2:
        diffusion_steps = diffusion_steps.squeeze(-1)

    half_dim = diffusion_step_embed_dim_in // 2

    # 计算频率：10000^(-2i/d)
    # 这使得低维度变化快，高维度变化慢，捕获不同频率的信息
    _embed = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=diffusion_steps.device) * -_embed)

    # 计算 step * 频率
    _embed = diffusion_steps.float()[:, None] * _embed[None, :]  # [B, half_dim]

    # 拼接sin和cos
    embed = torch.cat([torch.sin(_embed), torch.cos(_embed)], dim=1)  # [B, dim]

    return embed


class Conv1d(nn.Module):
    """
    带权重归一化的1D卷积层

    功能：
        标准的1D卷积，添加了权重归一化以提升训练稳定性
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv1d, self).__init__()

        # 计算padding以保持序列长度
        self.padding = dilation * (kernel_size - 1) // 2

        # 1D卷积
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding
        )

        # 权重归一化：提升训练稳定性
        self.conv = nn.utils.weight_norm(self.conv)

        # Kaiming初始化（He初始化）
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class ZeroConv1d(nn.Module):
    """
    零初始化的1D卷积层

    功能：
        权重和偏置都初始化为0的卷积层

    用途：
        在扩散模型中，零初始化的输出层可以使训练初期的预测接近原始输入
        有助于稳定训练过程
    """
    def __init__(self, in_channels, out_channels):
        super(ZeroConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

        # 零初始化
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block for Diffusion Model)

    功能：
        扩散模型的核心构建块，整合了：
        1. 扩散步骤嵌入
        2. 隐式和显式特征条件信息
        3. 扩张卷积
        4. 残差连接

    结构：
        输入 -> [+扩散嵌入] -> 扩张卷积 -> [+条件信息] -> 门控 -> [残差连接] -> 输出

    参数：
        res_channels (int): 残差层通道数
        skip_channels (int): 跳跃连接通道数
        dilation (int): 扩张率
        diffusion_step_embed_dim_out (int): 扩散步骤嵌入输出维度
        cond_channels (int): 条件信息通道数
    """
    def __init__(
        self,
        res_channels,
        skip_channels,
        dilation,
        diffusion_step_embed_dim_out,
        cond_channels
    ):
        super(ResidualBlock, self).__init__()

        self.res_channels = res_channels

        # === 扩散步骤嵌入的全连接层 ===
        # 将扩散步骤嵌入映射到残差通道维度
        self.fc_diffusion = nn.Linear(diffusion_step_embed_dim_out, res_channels)

        # === 扩张卷积层 ===
        # 使用扩张卷积扩大感受野
        # 输出通道数是输入的2倍，用于门控机制（tanh和sigmoid）
        self.dilated_conv = Conv1d(
            res_channels,
            2 * res_channels,
            kernel_size=3,
            dilation=dilation
        )

        # === 条件信息卷积层 ===
        # 将条件信息（隐式+显式特征）映射到适合门控的维度
        self.cond_conv = Conv1d(cond_channels, 2 * res_channels, kernel_size=1)

        # === 残差连接卷积层 ===
        # 1x1卷积用于调整残差连接
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # === 跳跃连接卷积层 ===
        # 1x1卷积用于跳跃连接
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, x, cond, diffusion_step_embed):
        """
        前向传播

        参数：
            x (torch.Tensor): 输入 [B, res_channels, L]
            cond (torch.Tensor): 条件信息 [B, cond_channels, L]
            diffusion_step_embed (torch.Tensor): 扩散步骤嵌入 [B, embed_dim]

        返回：
            residual_output (torch.Tensor): 残差输出 [B, res_channels, L]
            skip_output (torch.Tensor): 跳跃连接输出 [B, skip_channels, L]
        """
        B, C, L = x.shape
        assert C == self.res_channels

        # === 1. 添加扩散步骤嵌入 ===
        # 将嵌入映射到残差通道维度，然后广播到序列长度
        diffusion_embed = self.fc_diffusion(diffusion_step_embed)  # [B, res_channels]
        diffusion_embed = diffusion_embed.view(B, self.res_channels, 1)  # [B, res_channels, 1]
        h = x + diffusion_embed  # [B, res_channels, L]

        # === 2. 扩张卷积 ===
        h = self.dilated_conv(h)  # [B, 2*res_channels, L]

        # === 3. 添加条件信息 ===
        assert cond is not None, "条件信息不能为None"
        cond_out = self.cond_conv(cond)  # [B, 2*res_channels, L]
        h = h + cond_out  # [B, 2*res_channels, L]

        # === 4. 门控机制 (Gated Activation) ===
        # 将通道分为两半，分别应用tanh和sigmoid，然后相乘
        # 这种机制来自WaveNet，可以有效地控制信息流
        h_tanh = torch.tanh(h[:, :self.res_channels, :])  # [B, res_channels, L]
        h_sigmoid = torch.sigmoid(h[:, self.res_channels:, :])  # [B, res_channels, L]
        h = h_tanh * h_sigmoid  # [B, res_channels, L]

        # === 5. 残差连接和跳跃连接 ===
        residual_out = self.res_conv(h)  # [B, res_channels, L]
        skip_out = self.skip_conv(h)  # [B, skip_channels, L]

        # 残差连接：输出 = (输入 + 残差) * sqrt(0.5)
        # sqrt(0.5) 是缩放因子，用于训练稳定性
        residual_output = (x + residual_out) * math.sqrt(0.5)

        return residual_output, skip_out


class IEDiffusionModel(nn.Module):
    """
    隐式显式扩散模型 (Implicit-Explicit Diffusion Model)

    功能：
        完整的扩散模型，整合TCN隐式特征和S4显式特征，用于时间序列插补

    架构流程：
        1. DETACH: 输入分解为 (noise, observed_data, mask)
        2. 隐式提取: TCN处理输入，提取多时间尺度隐式特征
        3. 显式提取: S4处理输入，提取长期依赖显式特征
        4. 特征融合: 合并隐式和显式特征
        5. 条件准备: 将观测数据、mask和特征拼接作为条件
        6. 残差处理: 通过多个残差块处理，每块结合扩散步骤嵌入
        7. 输出生成: 聚合跳跃连接，生成去噪输出

    参数说明：
        in_channels (int): 输入通道数（传感器/特征数量）
        res_channels (int): 残差层通道数
        skip_channels (int): 跳跃连接通道数
        out_channels (int): 输出通道数（通常等于in_channels）
        num_res_layers (int): 残差层数量
        dilation_cycle (int): 扩张率循环周期

        diffusion_step_embed_dim_in (int): 扩散步骤嵌入输入维度
        diffusion_step_embed_dim_mid (int): 扩散步骤嵌入中间维度
        diffusion_step_embed_dim_out (int): 扩散步骤嵌入输出维度

        tcn_channels (list): TCN各层通道数
        tcn_kernel_size (int): TCN卷积核大小
        tcn_dilation_rates (list): TCN扩张率列表
        tcn_dropout (float): TCN的dropout

        s4_d_state (int): S4状态维度
        s4_n_layers (int): S4层数
        s4_l_max (int): S4最大序列长度
        s4_dropout (float): S4的dropout
        s4_bidirectional (bool): S4是否双向

    使用示例：
        ```python
        model = IEDiffusionModel(
            in_channels=370,
            res_channels=256,
            skip_channels=256,
            out_channels=370,
            num_res_layers=36,
            tcn_dilation_rates=[1, 2, 4, 8],
            s4_d_state=64,
            s4_bidirectional=True
        )
        ```
    """
    def __init__(
        self,
        in_channels,
        res_channels=256,
        skip_channels=256,
        out_channels=None,
        num_res_layers=36,
        dilation_cycle=10,
        # 扩散步骤嵌入参数
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
        # TCN隐式特征提取参数
        tcn_channels=[256, 256, 256],
        tcn_kernel_size=3,
        tcn_dilation_rates=[1, 2, 4, 8],
        tcn_dropout=0.0,
        # S4显式特征提取参数
        s4_d_state=64,
        s4_n_layers=4,
        s4_l_max=168,
        s4_dropout=0.0,
        s4_bidirectional=True
    ):
        super(IEDiffusionModel, self).__init__()

        # 如果未指定输出通道数，默认等于输入通道数
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.num_res_layers = num_res_layers

        # ====================================================================
        # 1. 隐式特征提取模块 (TCN)
        # ====================================================================
        self.tcn_extractor = TCNImplicitExtractor(
            in_channels=in_channels,
            hidden_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dilation_rates=tcn_dilation_rates,
            dropout=tcn_dropout
        )

        # ====================================================================
        # 2. 显式特征提取模块 (S4)
        # ====================================================================
        # S4需要特征维度作为输入，这里使用res_channels
        # 首先需要一个映射层将in_channels映射到res_channels
        self.input_projection = nn.Conv1d(in_channels, res_channels, kernel_size=1)

        self.s4_extractor = S4ExplicitExtractor(
            d_model=res_channels,
            d_state=s4_d_state,
            n_layers=s4_n_layers,
            l_max=s4_l_max,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional
        )

        # ====================================================================
        # 3. 初始卷积层
        # ====================================================================
        # 将噪声输入映射到残差通道维度
        self.init_conv = nn.Sequential(
            Conv1d(in_channels, res_channels, kernel_size=1),
            nn.ReLU()
        )

        # ====================================================================
        # 4. 扩散步骤嵌入网络
        # ====================================================================
        # 将扩散步骤编码映射到高维表示
        self.fc_diffusion1 = nn.Linear(
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid
        )
        self.fc_diffusion2 = nn.Linear(
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out
        )

        # ====================================================================
        # 5. 计算条件信息通道数
        # ====================================================================
        # 条件信息包括：observed_data (in_channels) + mask (in_channels)
        #               + implicit_features (1) + explicit_features (res_channels)
        cond_channels = in_channels + in_channels + 1 + res_channels

        # ====================================================================
        # 6. 残差块列表
        # ====================================================================
        self.residual_blocks = nn.ModuleList()
        for n in range(num_res_layers):
            # 计算扩张率：循环使用 1, 2, 4, ..., 2^(dilation_cycle-1)
            dilation = 2 ** (n % dilation_cycle)

            block = ResidualBlock(
                res_channels=res_channels,
                skip_channels=skip_channels,
                dilation=dilation,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                cond_channels=cond_channels
            )
            self.residual_blocks.append(block)

        # ====================================================================
        # 7. 输出层
        # ====================================================================
        # 将跳跃连接聚合后映射到输出
        self.final_conv = nn.Sequential(
            Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels)
        )

    def forward(self, noise, observed_data, mask, diffusion_steps):
        """
        前向传播

        参数：
            noise (torch.Tensor): 噪声输入 [B, in_channels, L]
            observed_data (torch.Tensor): 观测数据 [B, in_channels, L]
            mask (torch.Tensor): mask [B, in_channels, L]
                                1表示观测值，0表示缺失值
            diffusion_steps (torch.Tensor): 扩散步骤 [B] 或 [B, 1]

        返回：
            output (torch.Tensor): 去噪输出 [B, out_channels, L]

        处理流程：
            1. 提取隐式特征（TCN）
            2. 提取显式特征（S4）
            3. 准备条件信息
            4. 初始化噪声输入
            5. 计算扩散步骤嵌入
            6. 通过残差块处理
            7. 聚合跳跃连接
            8. 生成最终输出
        """
        B, C, L = noise.shape

        # ====================================================================
        # 步骤1: 准备输入 - 将观测数据应用mask
        # ====================================================================
        # 只保留观测值，缺失位置为0
        masked_data = observed_data * mask  # [B, C, L]

        # ====================================================================
        # 步骤2: 提取隐式特征 (TCN)
        # ====================================================================
        # TCN从masked_data中提取多时间尺度的隐式特征
        implicit_features = self.tcn_extractor(masked_data)  # [B, 1, L]

        # ====================================================================
        # 步骤3: 提取显式特征 (S4)
        # ====================================================================
        # 首先映射到res_channels维度
        s4_input = self.input_projection(masked_data)  # [B, res_channels, L]
        # S4提取长期依赖的显式特征
        explicit_features = self.s4_extractor(s4_input)  # [B, res_channels, L]

        # ====================================================================
        # 步骤4: 准备条件信息
        # ====================================================================
        # 将所有条件信息拼接：observed_data, mask, implicit_features, explicit_features
        conditional = torch.cat([
            masked_data,         # [B, in_channels, L]
            mask.float(),        # [B, in_channels, L]
            implicit_features,   # [B, 1, L]
            explicit_features    # [B, res_channels, L]
        ], dim=1)  # [B, cond_channels, L]

        # ====================================================================
        # 步骤5: 初始化噪声输入
        # ====================================================================
        x = self.init_conv(noise)  # [B, res_channels, L]

        # ====================================================================
        # 步骤6: 计算扩散步骤嵌入
        # ====================================================================
        # 将扩散步骤编码为高维向量
        diffusion_embed = calc_diffusion_step_embedding(
            diffusion_steps,
            self.fc_diffusion1.in_features
        )  # [B, diffusion_step_embed_dim_in]

        # 通过全连接层映射
        diffusion_embed = swish(self.fc_diffusion1(diffusion_embed))  # [B, mid_dim]
        diffusion_embed = swish(self.fc_diffusion2(diffusion_embed))  # [B, out_dim]

        # ====================================================================
        # 步骤7: 通过残差块处理
        # ====================================================================
        skip_sum = 0  # 累加所有跳跃连接

        for block in self.residual_blocks:
            # 每个残差块处理当前表示，并输出跳跃连接
            x, skip = block(x, conditional, diffusion_embed)
            skip_sum = skip_sum + skip

        # 归一化跳跃连接：除以残差块数量的平方根，用于训练稳定性
        skip_sum = skip_sum / math.sqrt(self.num_res_layers)

        # ====================================================================
        # 步骤8: 生成最终输出
        # ====================================================================
        output = self.final_conv(skip_sum)  # [B, out_channels, L]

        return output

    def get_config(self):
        """
        获取模型配置信息

        返回：
            config (dict): 包含模型配置的字典
        """
        return {
            'in_channels': self.in_channels,
            'res_channels': self.res_channels,
            'skip_channels': self.skip_channels,
            'out_channels': self.out_channels,
            'num_res_layers': self.num_res_layers,
            'tcn_config': self.tcn_extractor.get_config(),
            's4_config': self.s4_extractor.get_config(),
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        }


# ============================================================================
# 使用示例和调试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试隐式显式扩散模型
    """
    print("=" * 80)
    print("隐式显式扩散模型测试")
    print("=" * 80)

    # 设置随机种子以便复现
    torch.manual_seed(42)

    # 模拟数据
    batch_size = 4
    in_channels = 370   # 370个传感器
    seq_len = 168       # 168小时（1周）

    # 创建随机数据
    noise = torch.randn(batch_size, in_channels, seq_len)
    observed_data = torch.randn(batch_size, in_channels, seq_len)

    # 创建mask：30%缺失
    mask = torch.rand(batch_size, in_channels, seq_len) > 0.3
    mask = mask.float()

    # 扩散步骤：随机选择扩散步骤（0-199）
    diffusion_steps = torch.randint(0, 200, (batch_size,))

    print(f"\n输入形状:")
    print(f"  noise: {noise.shape}")
    print(f"  observed_data: {observed_data.shape}")
    print(f"  mask: {mask.shape} (缺失率: {1 - mask.mean():.2%})")
    print(f"  diffusion_steps: {diffusion_steps.shape}")

    # 创建模型
    model = IEDiffusionModel(
        in_channels=in_channels,
        res_channels=256,
        skip_channels=256,
        out_channels=in_channels,
        num_res_layers=36,
        # TCN参数
        tcn_channels=[256, 256, 256],
        tcn_dilation_rates=[1, 2, 4, 8],
        # S4参数
        s4_d_state=64,
        s4_n_layers=4,
        s4_l_max=seq_len,
        s4_bidirectional=True
    )

    print(f"\n模型创建完成！")

    # 前向传播
    print(f"\n执行前向传播...")
    output = model(noise, observed_data, mask, diffusion_steps)
    print(f"输出形状: {output.shape}")

    # 显示模型配置
    config = model.get_config()
    print(f"\n模型配置:")
    print(f"  输入通道数: {config['in_channels']}")
    print(f"  残差通道数: {config['res_channels']}")
    print(f"  残差层数: {config['num_res_layers']}")
    print(f"  TCN感受野: {config['tcn_config']['receptive_field']} 时间步")
    print(f"  S4状态维度: {config['s4_config']['d_state']}")
    print(f"  S4层数: {config['s4_config']['n_layers']}")
    print(f"  参数总数: {config['num_parameters']:,}")
    print(f"  可训练参数: {config['num_trainable_parameters']:,}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
