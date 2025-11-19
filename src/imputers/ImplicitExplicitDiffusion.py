"""
隐式-显式扩散模型 (Implicit-Explicit Diffusion Model)
用于交通流量数据插补

该模型结合了：
1. 隐式特征提取模块: 基于扩张因果卷积，捕获不同时间尺度的局部特征
2. 显式特征提取模块: 基于S4状态空间模型，捕获长期依赖关系
3. 扩散去噪模块: 基于DDPM的迭代去噪过程

作者: Claude AI
日期: 2025-11-19
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import calc_diffusion_step_embedding
from imputers.S4Model import S4Layer


def swish(x):
    """
    Swish激活函数 (也称为SiLU)
    f(x) = x * sigmoid(x)

    参数:
        x: 输入张量

    返回:
        激活后的张量
    """
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    """
    标准卷积层，带权重归一化和Kaiming初始化

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，默认为3
        dilation: 扩张率，用于控制感受野大小，默认为1
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        # 计算padding以保持序列长度不变
        # padding = dilation * (kernel_size - 1) // 2
        self.padding = dilation * (kernel_size - 1) // 2

        # 创建卷积层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             dilation=dilation, padding=self.padding)

        # 应用权重归一化以稳定训练
        self.conv = nn.utils.weight_norm(self.conv)

        # 使用Kaiming正态初始化权重
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, in_channels, length]

        返回:
            输出张量，形状为 [batch_size, out_channels, length]
        """
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    """
    零初始化的1x1卷积层
    用于模型输出层，确保初始时模型输出接近零
    这有助于稳定扩散模型的训练

    参数:
        in_channel: 输入通道数
        out_channel: 输出通道数
    """
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        # 创建1x1卷积
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)

        # 将权重和偏置初始化为零
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量

        返回:
            输出张量
        """
        out = self.conv(x)
        return out


class ImplicitFeatureExtractor(nn.Module):
    """
    隐式特征提取模块

    使用多尺度扩张因果卷积来捕获不同时间尺度的隐式特征
    通过不同的扩张率，模型可以同时捕获短期、中期和长期的时间依赖

    参数:
        channels: 通道数
        dilation_rates: 扩张率列表，例如[1, 2, 4, 8, 16]
                       不同的扩张率对应不同的时间尺度
                       - 小扩张率(1,2): 捕获短期依赖
                       - 中扩张率(4,8): 捕获中期依赖
                       - 大扩张率(16,32): 捕获长期依赖
        kernel_size: 卷积核大小
    """
    def __init__(self, channels, dilation_rates=[1, 2, 4, 8, 16], kernel_size=3):
        super(ImplicitFeatureExtractor, self).__init__()

        self.dilation_rates = dilation_rates
        self.num_layers = len(dilation_rates)

        # 创建多个扩张卷积层，每层对应一个扩张率
        self.dilated_convs = nn.ModuleList()
        for dilation in dilation_rates:
            # 每个扩张卷积层输出通道数与输入相同，便于特征融合
            self.dilated_convs.append(
                Conv(channels, channels, kernel_size=kernel_size, dilation=dilation)
            )

        # 特征融合层：将多尺度特征融合为统一维度
        # 使用1x1卷积进行通道融合
        self.fusion_conv = nn.Conv1d(
            channels * self.num_layers,  # 输入是所有尺度特征的拼接
            channels,                     # 输出恢复到原始通道数
            kernel_size=1
        )

        # 层归一化，提高训练稳定性
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, channels, length]

        返回:
            multi_scale_features: 融合后的多尺度隐式特征
                                 形状为 [batch_size, channels, length]
        """
        # 存储不同尺度的特征
        multi_scale_features = []

        # 通过不同扩张率的卷积提取不同尺度的特征
        for i, conv in enumerate(self.dilated_convs):
            # 应用扩张卷积
            feature = conv(x)
            # 应用激活函数
            feature = torch.tanh(feature)
            multi_scale_features.append(feature)

        # 沿通道维度拼接所有尺度的特征
        # 形状: [batch_size, channels * num_layers, length]
        concat_features = torch.cat(multi_scale_features, dim=1)

        # 通过1x1卷积融合多尺度特征
        # 形状: [batch_size, channels, length]
        fused_features = self.fusion_conv(concat_features)

        # 应用层归一化
        # 需要转置以符合LayerNorm的输入要求 [batch, length, channels]
        fused_features = fused_features.permute(0, 2, 1)
        fused_features = self.layer_norm(fused_features)
        fused_features = fused_features.permute(0, 2, 1)

        return fused_features


class ExplicitFeatureExtractor(nn.Module):
    """
    显式特征提取模块

    使用S4 (Structured State Space) 模型捕获长期时序依赖
    S4模型通过状态空间表示能够高效地建模长序列依赖关系

    参数:
        channels: 通道数
        s4_lmax: S4模型的最大序列长度
        s4_d_state: S4状态空间的维度（对应论文中的N）
                   更大的N可以捕获更复杂的动态模式，但计算成本更高
        s4_dropout: Dropout率，用于正则化
        s4_bidirectional: 是否使用双向S4
                         双向可以同时利用过去和未来的信息
        s4_layernorm: 是否使用层归一化
    """
    def __init__(self, channels, s4_lmax, s4_d_state, s4_dropout,
                 s4_bidirectional, s4_layernorm):
        super(ExplicitFeatureExtractor, self).__init__()

        # S4层用于捕获长期依赖
        # features参数是输入特征维度
        self.s4_layer = S4Layer(
            features=channels,
            lmax=s4_lmax,           # 最大序列长度
            N=s4_d_state,            # 状态空间维度
            dropout=s4_dropout,      # Dropout率
            bidirectional=s4_bidirectional,  # 是否双向
            layer_norm=s4_layernorm  # 是否使用层归一化
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, channels, length]

        返回:
            显式特征，形状为 [batch_size, channels, length]
        """
        # S4Layer期望输入形状为 [length, batch_size, channels]
        # 所以需要先转置
        x = x.permute(2, 0, 1)  # [length, batch, channels]

        # 通过S4层提取长期依赖特征
        x = self.s4_layer(x)    # [length, batch, channels]

        # 转回原始维度顺序 [batch, channels, length]
        x = x.permute(1, 2, 0)

        return x


class ImplicitExplicitResidualBlock(nn.Module):
    """
    隐式-显式残差块

    这是模型的核心组件，融合了：
    1. 隐式特征提取（多尺度扩张卷积）
    2. 显式特征提取（S4）
    3. 条件信息（观测数据和掩码）
    4. 扩散时间步嵌入

    参数:
        res_channels: 残差通道数
        skip_channels: 跳跃连接通道数
        diffusion_step_embed_dim_out: 扩散步嵌入的输出维度
        in_channels: 输入通道数
        implicit_dilation_rates: 隐式模块的扩张率列表
        s4_lmax: S4的最大序列长度
        s4_d_state: S4状态维度
        s4_dropout: S4的dropout率
        s4_bidirectional: S4是否双向
        s4_layernorm: S4是否使用层归一化
    """
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 implicit_dilation_rates,
                 s4_lmax, s4_d_state, s4_dropout,
                 s4_bidirectional, s4_layernorm):
        super(ImplicitExplicitResidualBlock, self).__init__()

        self.res_channels = res_channels

        # 扩散时间步嵌入的全连接层
        # 将时间步嵌入投影到残差通道维度
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # 隐式特征提取模块（多尺度扩张卷积）
        self.implicit_extractor = ImplicitFeatureExtractor(
            channels=self.res_channels,
            dilation_rates=implicit_dilation_rates
        )

        # 显式特征提取模块（S4）
        self.explicit_extractor = ExplicitFeatureExtractor(
            channels=self.res_channels,
            s4_lmax=s4_lmax,
            s4_d_state=s4_d_state,
            s4_dropout=s4_dropout,
            s4_bidirectional=s4_bidirectional,
            s4_layernorm=s4_layernorm
        )

        # 特征融合层
        # 将隐式和显式特征融合（通过拼接后的卷积）
        self.feature_fusion = Conv(
            2 * self.res_channels,  # 隐式 + 显式特征拼接
            2 * self.res_channels,   # 输出维度
            kernel_size=1
        )

        # 条件信息处理层（处理观测数据和掩码）
        self.cond_conv = Conv(
            2 * in_channels,         # 观测数据 + 掩码
            2 * self.res_channels,   # 输出维度
            kernel_size=1
        )

        # 额外的S4层，用于在加入条件信息后进一步处理
        # 这给模型额外的灵活性来融合特征和条件
        self.post_cond_s4 = S4Layer(
            features=2 * self.res_channels,
            lmax=s4_lmax,
            N=s4_d_state,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_layernorm
        )

        # 残差连接的卷积层
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # 跳跃连接的卷积层
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        """
        前向传播

        参数:
            input_data: 元组 (x, cond, diffusion_step_embed)
                x: 当前的噪声/数据，形状 [batch, res_channels, length]
                cond: 条件信息（观测+掩码），形状 [batch, 2*in_channels, length]
                diffusion_step_embed: 扩散步嵌入，形状 [batch, embed_dim]

        返回:
            (残差输出, 跳跃连接输出)
            残差输出形状: [batch, res_channels, length]
            跳跃连接输出形状: [batch, skip_channels, length]
        """
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        # 1. 添加扩散时间步信息
        # 将时间步嵌入投影并reshape为可以广播的形状
        part_t = self.fc_t(diffusion_step_embed)           # [B, res_channels]
        part_t = part_t.view([B, self.res_channels, 1])    # [B, res_channels, 1]
        h = h + part_t  # 广播加法，将时间步信息加到所有时间位置

        # 2. 隐式特征提取（多尺度扩张卷积）
        implicit_features = self.implicit_extractor(h)  # [B, res_channels, L]

        # 3. 显式特征提取（S4）
        explicit_features = self.explicit_extractor(h)  # [B, res_channels, L]

        # 4. 融合隐式和显式特征
        # 沿通道维度拼接
        combined_features = torch.cat([implicit_features, explicit_features], dim=1)
        # [B, 2*res_channels, L]

        # 通过卷积进一步融合
        h = self.feature_fusion(combined_features)  # [B, 2*res_channels, L]

        # 5. 加入条件信息（观测数据和掩码）
        assert cond is not None
        cond = self.cond_conv(cond)  # [B, 2*res_channels, L]
        h = h + cond  # 残差连接条件信息

        # 6. 通过S4层进一步处理融合后的特征
        # 转置以匹配S4Layer的输入格式
        h = h.permute(2, 0, 1)  # [L, B, 2*res_channels]
        h = self.post_cond_s4(h)  # [L, B, 2*res_channels]
        h = h.permute(1, 2, 0)  # [B, 2*res_channels, L]

        # 7. 门控激活（类似WaveNet）
        # 将特征分为两半，一半通过tanh，一半通过sigmoid，然后相乘
        out = torch.tanh(h[:, :self.res_channels, :]) * \
              torch.sigmoid(h[:, self.res_channels:, :])
        # [B, res_channels, L]

        # 8. 残差连接
        res = self.res_conv(out)
        assert x.shape == res.shape

        # 9. 跳跃连接
        skip = self.skip_conv(out)

        # 返回归一化的残差和跳跃连接
        # 乘以sqrt(0.5)是为了保持方差稳定
        return (x + res) * math.sqrt(0.5), skip


class ImplicitExplicitResidualGroup(nn.Module):
    """
    隐式-显式残差块组

    包含多个残差块的堆叠，以及扩散步嵌入的处理

    参数:
        res_channels: 残差通道数
        skip_channels: 跳跃连接通道数
        num_res_layers: 残差块数量，更多层可以建模更复杂的模式
        diffusion_step_embed_dim_in: 扩散步嵌入输入维度
        diffusion_step_embed_dim_mid: 扩散步嵌入中间维度
        diffusion_step_embed_dim_out: 扩散步嵌入输出维度
        in_channels: 输入通道数
        implicit_dilation_rates: 隐式模块的扩张率列表
        s4_lmax: S4最大序列长度
        s4_d_state: S4状态维度
        s4_dropout: S4 dropout率
        s4_bidirectional: S4是否双向
        s4_layernorm: S4是否使用层归一化
    """
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 implicit_dilation_rates,
                 s4_lmax, s4_d_state, s4_dropout,
                 s4_bidirectional, s4_layernorm):
        super(ImplicitExplicitResidualGroup, self).__init__()

        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # 扩散步嵌入的MLP（两层全连接网络）
        # 将扩散步骤t映射到高维嵌入空间
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        # 创建多个残差块
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                ImplicitExplicitResidualBlock(
                    res_channels=res_channels,
                    skip_channels=skip_channels,
                    diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                    in_channels=in_channels,
                    implicit_dilation_rates=implicit_dilation_rates,
                    s4_lmax=s4_lmax,
                    s4_d_state=s4_d_state,
                    s4_dropout=s4_dropout,
                    s4_bidirectional=s4_bidirectional,
                    s4_layernorm=s4_layernorm
                )
            )

    def forward(self, input_data):
        """
        前向传播

        参数:
            input_data: 元组 (noise, conditional, diffusion_steps)
                noise: 噪声数据，形状 [batch, res_channels, length]
                conditional: 条件信息，形状 [batch, 2*in_channels, length]
                diffusion_steps: 扩散步骤，形状 [batch, 1]

        返回:
            所有跳跃连接的加权和，形状 [batch, skip_channels, length]
        """
        noise, conditional, diffusion_steps = input_data

        # 1. 计算扩散步嵌入
        # 使用正弦/余弦位置编码将步骤t嵌入到高维空间
        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps,
            self.diffusion_step_embed_dim_in
        )

        # 2. 通过两层MLP处理嵌入，每层后使用swish激活
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # 3. 通过所有残差块
        h = noise
        skip = 0  # 累积所有跳跃连接

        for n in range(self.num_res_layers):
            # 每个残差块返回 (残差输出, 跳跃连接)
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))
            skip += skip_n  # 累加跳跃连接

        # 4. 归一化跳跃连接输出
        # 除以sqrt(num_layers)保持方差稳定
        return skip * math.sqrt(1.0 / self.num_res_layers)


class ImplicitExplicitDiffusion(nn.Module):
    """
    隐式-显式扩散模型主类

    这是完整的时序数据插补模型，结合了：
    1. 隐式特征提取：通过多尺度扩张卷积捕获局部模式
    2. 显式特征提取：通过S4捕获长期依赖
    3. 扩散去噪：通过DDPM迭代去噪恢复缺失数据

    模型架构：
    输入 -> 初始卷积 -> 残差块组（隐式+显式特征提取） -> 最终卷积 -> 输出

    参数说明：
    =========
    基本参数：
        in_channels: 输入通道数（数据特征维度）
        res_channels: 残差连接通道数，控制模型容量
        skip_channels: 跳跃连接通道数
        out_channels: 输出通道数（通常等于in_channels）
        num_res_layers: 残差块数量，更多层可以建模更复杂的模式

    扩散相关参数：
        diffusion_step_embed_dim_in: 扩散步嵌入输入维度
        diffusion_step_embed_dim_mid: 扩散步嵌入中间维度
        diffusion_step_embed_dim_out: 扩散步嵌入输出维度

    隐式模块参数（扩张卷积）：
        implicit_dilation_rates: 扩张率列表，例如[1,2,4,8,16]
                                控制不同时间尺度的特征提取
                                - 调整此参数可以改变隐式模块捕获的时间尺度范围
                                - 增加扩张率可以捕获更长期的依赖
                                - 减少扩张率专注于短期依赖

    显式模块参数（S4状态空间模型）：
        s4_lmax: S4最大序列长度，应该>=实际序列长度
        s4_d_state: S4状态空间维度（论文中的N）
                   - 对应S4的A,B,C,D矩阵的维度
                   - 更大的N可以捕获更复杂的动态，但计算成本更高
                   - 典型值：64, 128, 256
        s4_dropout: S4层的dropout率，用于正则化
        s4_bidirectional: 是否使用双向S4
                         - True: 可以利用未来信息，适合离线任务
                         - False: 只使用过去信息，适合在线任务
        s4_layernorm: 是否在S4层使用层归一化

    使用示例：
    =========
    # 创建模型
    model = ImplicitExplicitDiffusion(
        in_channels=14,           # 14个交通传感器
        res_channels=256,         # 残差通道数
        skip_channels=256,        # 跳跃连接通道数
        out_channels=14,          # 输出14个传感器的预测
        num_res_layers=36,        # 36个残差块
        implicit_dilation_rates=[1, 2, 4, 8, 16],  # 5个不同时间尺度
        s4_lmax=100,              # 最大序列长度100
        s4_d_state=64,            # 状态维度64
        ...
    )

    # 前向传播
    output = model((noise, observed_data, mask, diffusion_steps))
    """
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 implicit_dilation_rates,
                 s4_lmax, s4_d_state, s4_dropout,
                 s4_bidirectional, s4_layernorm):
        super(ImplicitExplicitDiffusion, self).__init__()

        # 初始卷积层：将输入投影到残差通道空间
        # 使用1x1卷积 + ReLU激活
        self.init_conv = nn.Sequential(
            Conv(in_channels, res_channels, kernel_size=1),
            nn.ReLU()
        )

        # 主要的残差块组
        # 这是模型的核心，包含所有隐式-显式特征提取逻辑
        self.residual_layer = ImplicitExplicitResidualGroup(
            res_channels=res_channels,
            skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
            in_channels=in_channels,
            implicit_dilation_rates=implicit_dilation_rates,
            s4_lmax=s4_lmax,
            s4_d_state=s4_d_state,
            s4_dropout=s4_dropout,
            s4_bidirectional=s4_bidirectional,
            s4_layernorm=s4_layernorm
        )

        # 最终卷积层：将跳跃连接特征投影到输出空间
        # 使用两层1x1卷积，最后一层是零初始化
        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),  # 特征变换
            nn.ReLU(),                                          # 激活
            ZeroConv1d(skip_channels, out_channels)            # 零初始化输出
        )

    def forward(self, input_data):
        """
        前向传播

        参数:
            input_data: 元组 (noise, conditional, mask, diffusion_steps)
                noise: 当前的噪声数据，形状 [batch, in_channels, length]
                       在训练时是加噪的数据，推理时初始为高斯噪声
                conditional: 观测数据（条件），形状 [batch, in_channels, length]
                            只有观测位置有值，缺失位置为0
                mask: 二值掩码，形状 [batch, in_channels, length]
                     1表示观测位置，0表示缺失位置
                diffusion_steps: 当前扩散步骤，形状 [batch, 1]
                                取值范围 [0, T-1]，其中T是总扩散步数

        返回:
            预测的噪声，形状 [batch, out_channels, length]
            在训练时用于与真实噪声计算损失
            在推理时用于去噪

        处理流程:
            1. 将观测数据和掩码拼接作为条件信息
            2. 通过初始卷积处理噪声数据
            3. 通过残差块组提取隐式和显式特征
            4. 通过最终卷积生成噪声预测
        """
        noise, conditional, mask, diffusion_steps = input_data

        # 1. 准备条件信息
        # 将观测数据乘以掩码，确保缺失位置为0
        conditional = conditional * mask

        # 将观测数据和掩码沿通道维度拼接
        # 形状: [batch, 2*in_channels, length]
        # 这样模型可以同时知道观测值和观测位置
        conditional = torch.cat([conditional, mask.float()], dim=1)

        # 2. 初始特征提取
        x = noise
        x = self.init_conv(x)  # [batch, res_channels, length]

        # 3. 通过残差块组提取隐式-显式特征
        # 返回所有跳跃连接的累积和
        x = self.residual_layer((x, conditional, diffusion_steps))
        # [batch, skip_channels, length]

        # 4. 生成最终输出（噪声预测）
        y = self.final_conv(x)  # [batch, out_channels, length]

        return y
