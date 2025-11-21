"""
TCN隐式特征提取模块 (Temporal Convolutional Network for Implicit Feature Extraction)

功能说明：
    使用扩张因果卷积提取时间序列的隐式特征，捕获不同时间尺度的依赖关系

核心概念：
    - 因果卷积：保证时间t的输出只依赖于t及之前的输入，不会产生信息泄露
    - 扩张卷积：通过调整扩张率(dilation rate)来扩大感受野，捕获更长时间尺度的依赖
    - 多尺度特征：不同扩张率的卷积层可以提取不同时间尺度的特征

超参数调节指南：
    1. 基础超参数：
       - channels: 每层的通道数，增大可提升表达能力但会增加计算量 [建议: 64-256]
       - kernel_size: 卷积核大小，影响局部感受野 [建议: 3]
       - dropout: 防止过拟合的dropout比率 [建议: 0.0-0.2]

    2. 架构调整：
       - dilation_rates: 扩张率序列，控制时间尺度
         * [1, 2, 4, 8]: 标准指数增长，捕获1到8步的依赖
         * [1, 2, 4, 8, 16]: 更长的依赖关系
         * [1, 4, 16]: 跳跃式增长，捕获更稀疏的长期依赖
         * 计算感受野: RF = 1 + 2 * (kernel_size - 1) * sum(dilation_rates)

       - num_layers: TCN层数，增加深度可以提取更抽象的特征 [建议: 3-6]

    3. 优化技巧：
       - 使用weight normalization提升训练稳定性
       - 使用residual connection加速收敛
       - 使用dropout防止过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    因果卷积层 (Causal Convolution Layer)

    特点：
        - 保证因果性：输出只依赖于当前和过去的输入
        - 使用padding确保输出长度与输入相同

    参数说明：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，通常设为3
        dilation (int): 扩张率，控制卷积核元素之间的间隔
            - dilation=1: 标准卷积，感受野为kernel_size
            - dilation=2: 每隔一个位置采样，感受野扩大2倍
            - dilation=4: 每隔三个位置采样，感受野扩大4倍
        dropout (float): Dropout比率，用于正则化

    输入输出：
        输入: [batch_size, in_channels, seq_len]
        输出: [batch_size, out_channels, seq_len]
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(CausalConv1d, self).__init__()

        # 计算因果卷积所需的padding
        # padding = (kernel_size - 1) * dilation 确保输出长度等于输入长度
        # 这个padding会添加在序列的左侧（过去），保证因果性
        self.padding = (kernel_size - 1) * dilation

        # 一维卷积层
        # dilation参数控制卷积核元素之间的间隔，实现不同时间尺度的特征提取
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # 我们手动处理padding以实现因果性
            dilation=dilation
        )

        # Weight Normalization: 将权重向量分解为方向和幅度
        # 优点：加速收敛，提高训练稳定性，对初始化不敏感
        self.conv = nn.utils.weight_norm(self.conv)

        # Dropout: 训练时随机丢弃一些神经元，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 使用Kaiming初始化（He初始化）
        # 适用于ReLU激活函数，有助于避免梯度消失/爆炸
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 [batch_size, in_channels, seq_len]

        返回：
            out: 输出张量 [batch_size, out_channels, seq_len]
        """
        # 在序列左侧（过去）添加padding，保证因果性
        # F.pad的参数(self.padding, 0)表示在最后一维的左侧填充self.padding个0，右侧不填充
        x = F.pad(x, (self.padding, 0))

        # 执行卷积操作
        x = self.conv(x)

        # 应用dropout
        x = self.dropout(x)

        return x


class TCNResidualBlock(nn.Module):
    """
    TCN残差块 (TCN Residual Block)

    结构：
        输入 -> CausalConv1d -> ReLU -> CausalConv1d -> ReLU -> 输出
         |                                                      |
         +-------------------residual connection----------------+

    优点：
        - 残差连接允许梯度直接传播，解决深层网络训练困难的问题
        - 使网络能够学习残差（增量），而不是完整的变换
        - 允许堆叠更多层以提取更抽象的特征

    参数说明：
        channels (int): 输入输出通道数（残差块保持通道数不变）
        kernel_size (int): 卷积核大小
        dilation (int): 扩张率
        dropout (float): Dropout比率
    """
    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super(TCNResidualBlock, self).__init__()

        # 第一个因果卷积层
        # 使用相同的dilation rate，保持时间尺度一致
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size, dilation, dropout
        )

        # 第二个因果卷积层
        # 进一步提取特征，增加网络的非线性表达能力
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size, dilation, dropout
        )

        # ReLU激活函数：max(0, x)
        # 引入非线性，使网络能够拟合复杂函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 [batch_size, channels, seq_len]

        返回：
            out: 输出张量 [batch_size, channels, seq_len]
        """
        # 保存输入，用于残差连接
        residual = x

        # 第一个卷积 -> 激活
        out = self.relu(self.conv1(x))

        # 第二个卷积 -> 激活
        out = self.relu(self.conv2(out))

        # 残差连接：输出 = F(x) + x
        # 这使得网络学习残差F(x)而不是完整的映射H(x)
        # 如果F(x)=0，则网络退化为恒等映射，不会降低性能
        out = out + residual

        return out


class TCNImplicitExtractor(nn.Module):
    """
    TCN隐式特征提取器 (TCN Implicit Feature Extractor)

    功能：
        通过堆叠多个不同扩张率的TCN残差块，提取多时间尺度的隐式特征

    架构设计：
        输入 -> 输入卷积 -> [TCN残差块1, TCN残差块2, ..., TCN残差块N] -> 输出卷积 -> 隐式特征

    多尺度特征提取原理：
        - 第1层 (dilation=1): 捕获相邻时间步的局部特征
        - 第2层 (dilation=2): 捕获间隔1步的短期依赖
        - 第3层 (dilation=4): 捕获间隔3步的中期依赖
        - 第4层 (dilation=8): 捕获间隔7步的长期依赖

    参数说明：
        in_channels (int): 输入通道数，对应时间序列的特征维度
        hidden_channels (list): 每层TCN的通道数列表
            - 例如 [64, 64, 64] 表示3层，每层64个通道
            - 可以设置为递增 [64, 128, 256] 以逐层提取更抽象的特征
        kernel_size (int): 卷积核大小，通常设为3
        dilation_rates (list): 扩张率列表，每个元素对应一个TCN层
            - [1, 2, 4, 8]: 指数增长，适合大多数时间序列
            - [1, 4, 16]: 跳跃增长，适合需要捕获稀疏长期依赖的情况
        dropout (float): Dropout比率

    调优建议：
        1. 如果模型过拟合：增大dropout (0.1 -> 0.2)
        2. 如果需要更长的依赖：增加dilation_rates (添加16, 32等)
        3. 如果特征不够丰富：增加hidden_channels (64 -> 128)
        4. 如果训练不稳定：减小学习率或增加batch normalization
    """
    def __init__(
        self,
        in_channels,
        hidden_channels=[256, 256, 256],
        kernel_size=3,
        dilation_rates=[1, 2, 4, 8],
        dropout=0.0
    ):
        super(TCNImplicitExtractor, self).__init__()

        # 输入卷积：将输入维度映射到隐藏维度
        # 使用1x1卷积（kernel_size=1）进行通道数变换，不改变序列长度
        self.input_conv = nn.Conv1d(in_channels, hidden_channels[0], kernel_size=1)
        self.input_conv = nn.utils.weight_norm(self.input_conv)
        nn.init.kaiming_normal_(self.input_conv.weight)

        # TCN残差块列表
        self.tcn_blocks = nn.ModuleList()

        # 为每个扩张率创建一个TCN残差块
        # 如果hidden_channels长度小于dilation_rates，则重复使用最后一个channel数
        for i, dilation in enumerate(dilation_rates):
            # 确定当前层的通道数
            if i < len(hidden_channels):
                channels = hidden_channels[i]
            else:
                channels = hidden_channels[-1]

            # 如果不是第一层，且通道数改变，需要添加通道变换层
            if i > 0 and channels != self.tcn_blocks[-1].conv1.conv.out_channels:
                # 添加1x1卷积进行通道数变换
                channel_transform = nn.Conv1d(
                    self.tcn_blocks[-1].conv1.conv.out_channels,
                    channels,
                    kernel_size=1
                )
                channel_transform = nn.utils.weight_norm(channel_transform)
                self.tcn_blocks.append(channel_transform)

            # 添加TCN残差块
            # 每个块使用不同的扩张率，捕获不同时间尺度的特征
            block = TCNResidualBlock(
                channels,
                kernel_size,
                dilation,
                dropout
            )
            self.tcn_blocks.append(block)

        # 输出卷积：将隐藏维度映射到输出维度
        # 这里输出通道数设为1，用于生成标量形式的隐式特征
        final_channels = hidden_channels[-1] if hidden_channels else hidden_channels[0]
        self.output_conv = nn.Conv1d(final_channels, 1, kernel_size=1)
        self.output_conv = nn.utils.weight_norm(self.output_conv)
        nn.init.kaiming_normal_(self.output_conv.weight)

        # 保存配置用于调试和模型分析
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dilation_rates = dilation_rates

        # 计算理论感受野（Receptive Field）
        # 感受野表示输出神经元能"看到"的输入范围
        self.receptive_field = self._calculate_receptive_field(kernel_size, dilation_rates)

    def _calculate_receptive_field(self, kernel_size, dilation_rates):
        """
        计算TCN的理论感受野

        感受野计算公式：
            RF = 1 + 2 * (kernel_size - 1) * sum(dilation_rates)

        例如：kernel_size=3, dilation_rates=[1,2,4,8]
            RF = 1 + 2 * (3-1) * (1+2+4+8) = 1 + 2*2*15 = 61

        这意味着输出的每个位置可以"看到"过去61个时间步的信息
        """
        rf = 1
        for dilation in dilation_rates:
            rf += 2 * (kernel_size - 1) * dilation
        return rf

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 [batch_size, in_channels, seq_len]
               - batch_size: 批次大小
               - in_channels: 输入特征维度（例如交通传感器数量）
               - seq_len: 序列长度（例如时间步数）

        返回：
            implicit_features: 隐式特征 [batch_size, 1, seq_len]
                             或 [batch_size, hidden_channels[-1], seq_len]

        处理流程：
            1. 输入卷积：调整通道数
            2. 依次通过所有TCN残差块：提取多尺度特征
            3. 输出卷积：生成最终的隐式特征表示
        """
        # 输入卷积：[B, in_channels, L] -> [B, hidden_channels[0], L]
        out = self.input_conv(x)

        # 通过所有TCN残差块
        # 每个块提取特定时间尺度的特征
        for block in self.tcn_blocks:
            out = block(out)

        # 输出卷积：生成隐式特征
        # [B, hidden_channels[-1], L] -> [B, 1, L] 或保持多通道
        implicit_features = self.output_conv(out)

        return implicit_features

    def get_config(self):
        """
        获取模型配置信息，用于模型分析和调试

        返回：
            config (dict): 包含模型配置的字典
        """
        return {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'dilation_rates': self.dilation_rates,
            'receptive_field': self.receptive_field,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# 使用示例和调试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试TCN隐式特征提取器

    这个测试代码展示了如何使用TCN模块，以及如何查看模型配置
    """
    print("=" * 80)
    print("TCN隐式特征提取器测试")
    print("=" * 80)

    # 模拟输入数据
    batch_size = 4      # 批次大小
    in_channels = 370   # 输入通道数（例如370个交通传感器）
    seq_len = 168       # 序列长度（例如168小时=1周）

    # 创建随机输入张量
    x = torch.randn(batch_size, in_channels, seq_len)
    print(f"\n输入张量形状: {x.shape}")

    # 创建TCN模型
    model = TCNImplicitExtractor(
        in_channels=in_channels,
        hidden_channels=[256, 256, 256],  # 3层，每层256通道
        kernel_size=3,
        dilation_rates=[1, 2, 4, 8],      # 4个不同的时间尺度
        dropout=0.1
    )

    # 前向传播
    output = model(x)
    print(f"输出张量形状: {output.shape}")

    # 显示模型配置
    config = model.get_config()
    print(f"\n模型配置:")
    print(f"  输入通道数: {config['in_channels']}")
    print(f"  隐藏层通道数: {config['hidden_channels']}")
    print(f"  扩张率: {config['dilation_rates']}")
    print(f"  感受野: {config['receptive_field']} 时间步")
    print(f"  参数总数: {config['num_parameters']:,}")
    print(f"  可训练参数: {config['num_trainable_parameters']:,}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
