# 隐式-显式扩散模型用于交通流量数据插补

## 目录
- [项目概述](#项目概述)
- [模型架构](#模型架构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [超参数调优指南](#超参数调优指南)
- [代码结构](#代码结构)
- [实验结果](#实验结果)
- [常见问题](#常见问题)

---

## 项目概述

本项目实现了一个**隐式-显式扩散模型**(Implicit-Explicit Diffusion Model)，用于交通流量数据的时间序列插补任务。该模型结合了：

1. **隐式特征提取模块**：基于多尺度扩张因果卷积，捕获不同时间尺度的局部特征
2. **显式特征提取模块**：基于S4状态空间模型，捕获长期时序依赖
3. **扩散去噪模块**：基于DDPM的迭代去噪过程，实现高质量的数据插补

### 主要特点

- ✅ 支持多种缺失率（20%-80%）的训练和评估
- ✅ 详细的代码注释，便于理解和修改
- ✅ 灵活的超参数配置
- ✅ 自动化的实验流程
- ✅ 完整的可视化和评估工具

---

## 模型架构

### 整体架构

```
输入序列
    ↓
DETACH模块：分解为 (noise, observed_data, mask, implicit_info, explicit_info)
    ↓
┌─────────────────┬─────────────────┐
│   隐式特征提取   │   显式特征提取   │
│  (扩张因果卷积)  │   (S4模型)      │
└─────────────────┴─────────────────┘
    ↓              ↓
    └──────┬───────┘
          ↓
    特征融合 + 条件信息
          ↓
    残差卷积层
          ↓
    扩散步嵌入
          ↓
    最终输出（噪声预测）
```

### 核心模块详解

#### 1. 隐式特征提取模块 (ImplicitFeatureExtractor)

**功能**：使用多尺度扩张因果卷积捕获不同时间尺度的特征

**关键参数**：
- `dilation_rates`: 扩张率列表，默认 `[1, 2, 4, 8, 16]`
  - `1`: 捕获1步局部依赖
  - `2`: 捕获2步依赖
  - `4`: 捕获4步依赖
  - `8`: 捕获8步依赖
  - `16`: 捕获16步依赖

**代码位置**：`src/imputers/ImplicitExplicitDiffusion.py` (第92-156行)

**如何调整**：
```python
# 修改扩张率以改变捕获的时间尺度
# 例如，增加更长期的依赖：
implicit_dilation_rates = [1, 2, 4, 8, 16, 32, 64]

# 或者只关注短期依赖：
implicit_dilation_rates = [1, 2, 4]
```

#### 2. 显式特征提取模块 (ExplicitFeatureExtractor)

**功能**：使用S4状态空间模型捕获长期依赖

**关键参数**：
- `s4_lmax`: 最大序列长度（应 >= 实际序列长度）
- `s4_d_state`: S4状态维度，对应论文中的N
  - 控制S4的A, B, C, D矩阵维度
  - 更大的N可以建模更复杂的动态
  - 典型值：32, 64, 128, 256
- `s4_bidirectional`: 是否双向
  - `True`: 利用过去和未来信息（离线任务）
  - `False`: 只利用过去信息（在线任务）

**代码位置**：`src/imputers/ImplicitExplicitDiffusion.py` (第159-196行)

**S4核心理论**：

S4模型基于状态空间表示：
```
dx(t)/dt = Ax(t) + Bu(t)    # 状态转移方程
y(t) = Cx(t) + Du(t)        # 观测方程
```

其中：
- A ∈ ℝ^(N×N): 状态转移矩阵
- B ∈ ℝ^(N×1): 输入矩阵
- C ∈ ℝ^(1×N): 输出矩阵
- D ∈ ℝ: 直接传递项

**如何调整S4参数**：
```python
# 在配置文件中修改：
{
    "s4_lmax": 100,        # 序列长度
    "s4_d_state": 64,       # 状态维度N
    "s4_dropout": 0.0,      # Dropout率
    "s4_bidirectional": 1,  # 1=双向，0=单向
    "s4_layernorm": 1       # 1=使用层归一化，0=不使用
}
```

#### 3. 扩散去噪模块

**功能**：通过DDPM扩散过程实现迭代去噪

**关键参数**：
- `T`: 扩散步数，默认200
  - 更多步数：去噪更细致，但推理更慢
  - 更少步数：推理更快，但质量可能下降
- `beta_0`: 噪声方差起始值，默认0.0001
- `beta_T`: 噪声方差结束值，默认0.02

**前向扩散过程**（训练时）：
```
q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
```

**反向去噪过程**（推理时）：
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t^2 I)
```

---

## 环境配置

### 硬件要求

- **GPU**: NVIDIA GPU with CUDA support
  - 推荐: A10, A40, V100, 或更高
  - 最小显存: 8GB
  - 推荐显存: 16GB+

### 软件依赖

```bash
# Python版本
Python >= 3.8

# 核心依赖
torch >= 1.10.0
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
scikit-learn >= 0.24.0
tqdm >= 4.62.0
scipy >= 1.7.0
opt_einsum >= 3.3.0
```

### 安装步骤

1. **创建虚拟环境**：
```bash
conda create -n implicit_explicit python=3.8
conda activate implicit_explicit
```

2. **安装PyTorch**（根据你的CUDA版本）：
```bash
# CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 或 CUDA 11.7
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. **安装其他依赖**：
```bash
cd /home/user/SSSD/src
pip install -r requirements.txt
```

4. **（可选）安装Cauchy扩展以加速S4**：
```bash
cd /home/user/SSSD/src/entensions/cauchy
python setup.py install
```

---

## 快速开始

### 方法1：使用自动化脚本（推荐）

```bash
cd /home/user/SSSD
./run_experiments.sh
```

这个脚本会自动执行：
1. 数据预处理
2. 在所有缺失率下训练模型（20%-80%）
3. 评估并生成报告

### 方法2：手动执行

#### 步骤1：数据预处理

```bash
cd /home/user/SSSD/src
python data_loader_ld2011.py
```

**输出**：
- `datasets/train_ld2011.npy`: 训练数据
- `datasets/test_ld2011.npy`: 测试数据

#### 步骤2：训练模型

训练单个缺失率：
```bash
# 20%缺失率
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0

# 50%缺失率
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.5 \
    --device cuda:0
```

训练所有缺失率：
```bash
for ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    python train_ld2011.py \
        --config config/config_ImplicitExplicit_LD2011.json \
        --missing_ratio $ratio \
        --device cuda:0
done
```

#### 步骤3：评估模型

评估所有缺失率：
```bash
python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --eval_all \
    --device cuda:0
```

评估单个缺失率：
```bash
python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

---

## 详细使用说明

### 配置文件详解

配置文件位于：`src/config/config_ImplicitExplicit_LD2011.json`

```json
{
    "diffusion_config": {
        "T": 200,              // 扩散步数（推荐: 100-500）
        "beta_0": 0.0001,      // 初始噪声方差（推荐: 0.0001-0.001）
        "beta_T": 0.02         // 最终噪声方差（推荐: 0.01-0.05）
    },
    "model_config": {
        // 基本设置
        "in_channels": 14,          // 输入特征数（LD2011数据集）
        "out_channels": 14,         // 输出特征数
        "num_res_layers": 36,       // 残差块数量（推荐: 20-50）
        "res_channels": 256,        // 残差通道数（推荐: 128-512）
        "skip_channels": 256,       // 跳跃连接通道数

        // 扩散步嵌入
        "diffusion_step_embed_dim_in": 128,   // 输入维度
        "diffusion_step_embed_dim_mid": 512,  // 中间维度
        "diffusion_step_embed_dim_out": 512,  // 输出维度

        // 隐式模块（扩张卷积）
        "implicit_dilation_rates": [1, 2, 4, 8, 16],  // 扩张率

        // 显式模块（S4）
        "s4_lmax": 100,             // 最大序列长度
        "s4_d_state": 64,           // 状态维度（推荐: 32-256）
        "s4_dropout": 0.0,          // Dropout率
        "s4_bidirectional": 1,      // 双向
        "s4_layernorm": 1           // 层归一化
    },
    "train_config": {
        "output_directory": "./results/ld2011",  // 输出目录
        "ckpt_iter": "max",                      // 检查点恢复
        "iters_per_ckpt": 1000,                  // 保存检查点频率
        "iters_per_logging": 100,                // 记录日志频率
        "n_iters": 50000,                        // 总迭代次数
        "learning_rate": 2e-4,                   // 学习率
        "only_generate_missing": 1,              // 只对缺失部分扩散
        "masking": "rm",                         // 缺失模式
        "missing_ratio": 0.2,                    // 缺失率
        "batch_size": 16,                        // 批次大小
        "device": "cuda:0"                       // GPU设备
    }
}
```

### 数据格式

#### 输入数据格式

LD2011_2014.txt文件格式：
- 分隔符：分号(;)、制表符(\t)或逗号(,)
- 第一行：列名（可选）
- 后续行：每行是一个时间步的数据

示例：
```
timestamp;sensor1;sensor2;sensor3;...;sensor14
2011-01-01 00:00:00;0.123;0.456;0.789;...;0.234
2011-01-01 00:15:00;0.234;0.567;0.890;...;0.345
...
```

#### 预处理后的数据格式

- 训练数据：`[样本数, 序列长度, 特征数]`
- 例如：`(500, 100, 14)` 表示500个样本，每个样本长度100，14个特征

---

## 超参数调优指南

### 初级：基本超参数调整

这些参数不涉及模型架构修改，适合初学者：

#### 1. 学习率 (learning_rate)

```json
"learning_rate": 2e-4  // 默认值
```

**调整建议**：
- 训练不稳定/损失爆炸：减小学习率（1e-4）
- 训练太慢：增大学习率（5e-4）
- 使用学习率衰减：
  ```python
  # 在train_ld2011.py中添加
  from torch.optim.lr_scheduler import CosineAnnealingLR
  scheduler = CosineAnnealingLR(optimizer, T_max=n_iters)
  ```

#### 2. 批次大小 (batch_size)

```json
"batch_size": 16  // 默认值
```

**调整建议**：
- 显存不足：减小批次大小（8, 4）
- 显存充足：增大批次大小（32, 64）
- 更大的批次可以加速训练但可能需要调整学习率

#### 3. 训练迭代次数 (n_iters)

```json
"n_iters": 50000  // 默认值
```

**调整建议**：
- 快速测试：5000-10000
- 正常训练：30000-50000
- 完整训练：100000+

#### 4. 扩散步数 (T)

```json
"T": 200  // 默认值
```

**调整建议**：
- 更快推理：减少步数（100）
- 更高质量：增加步数（500）

### 中级：架构参数调整

这些参数会改变模型容量：

#### 1. 残差通道数 (res_channels)

```json
"res_channels": 256  // 默认值
```

**调整建议**：
- 小模型（显存受限）：128
- 中模型：256
- 大模型：512

**影响**：
- 更大的通道数 → 更强的表达能力，但需要更多显存和计算

#### 2. 残差块数量 (num_res_layers)

```json
"num_res_layers": 36  // 默认值
```

**调整建议**：
- 简单任务：20-30
- 中等任务：30-40
- 复杂任务：40-50

**影响**：
- 更多层 → 更深的网络，可以建模更复杂的模式

### 高级：模块架构调整

这些参数会改变隐式/显式模块的架构：

#### 1. 隐式模块扩张率 (implicit_dilation_rates)

```json
"implicit_dilation_rates": [1, 2, 4, 8, 16]  // 默认值
```

**调整策略**：

**捕获更长期依赖**：
```json
"implicit_dilation_rates": [1, 2, 4, 8, 16, 32, 64]
```

**只关注短期模式**：
```json
"implicit_dilation_rates": [1, 2, 4, 8]
```

**非指数增长**：
```json
"implicit_dilation_rates": [1, 3, 5, 7, 9]
```

**理解扩张率**：
- 扩张率d=1：标准卷积，感受野=kernel_size
- 扩张率d=2：每次跳过1个位置，感受野扩大2倍
- 扩张率d=k：每次跳过k-1个位置

**可视化示例**（kernel_size=3）：
```
d=1: [x x x . . .]  感受野: 3
d=2: [x . x . x .]  感受野: 5
d=4: [x . . . x . . . x]  感受野: 9
```

#### 2. S4状态维度 (s4_d_state)

```json
"s4_d_state": 64  // 默认值
```

**调整策略**：

**更简单的动态**：
```json
"s4_d_state": 32
```

**更复杂的动态**：
```json
"s4_d_state": 128  // 或 256
```

**理解状态维度**：

S4的状态维度N控制了状态空间矩阵的大小：
- A: N×N （状态转移矩阵）
- B: N×1 （输入矩阵）
- C: 1×N （输出矩阵）

更大的N意味着：
- ✅ 可以建模更复杂的时序动态
- ✅ 更强的长期依赖建模能力
- ❌ 更多的参数量
- ❌ 更高的计算成本

**实验建议**：
- 从N=64开始
- 如果模型欠拟合（训练损失高），增加N
- 如果模型过拟合（训练损失低但测试差），减少N或增加正则化

#### 3. S4双向性 (s4_bidirectional)

```json
"s4_bidirectional": 1  // 1=双向，0=单向
```

**选择指南**：

**双向（bidirectional=1）**：
- 适用场景：离线数据插补、批量处理
- 优点：可以利用过去和未来的信息
- 缺点：不能用于实时在线预测

**单向（bidirectional=0）**：
- 适用场景：在线预测、实时系统
- 优点：只依赖历史信息，可以实时运行
- 缺点：没有未来信息，可能性能略差

#### 4. S4最大序列长度 (s4_lmax)

```json
"s4_lmax": 100  // 默认值
```

**调整建议**：
- 应该设置为 >= 实际序列长度
- 如果序列长度是100，设置s4_lmax=100或更大
- 更大的值会增加计算成本

### 实验流程建议

#### 第一阶段：快速验证

目标：验证代码能够运行，模型能够学习

```json
{
    "n_iters": 5000,
    "num_res_layers": 20,
    "res_channels": 128,
    "s4_d_state": 32,
    "implicit_dilation_rates": [1, 2, 4, 8]
}
```

#### 第二阶段：基线性能

目标：使用默认参数获得基线性能

```json
{
    "n_iters": 30000,
    "num_res_layers": 36,
    "res_channels": 256,
    "s4_d_state": 64,
    "implicit_dilation_rates": [1, 2, 4, 8, 16]
}
```

#### 第三阶段：超参数优化

基于基线结果，调整关键参数：

**如果基线欠拟合（训练损失高）**：
1. 增加模型容量：
   - `res_channels`: 256 → 512
   - `num_res_layers`: 36 → 48
   - `s4_d_state`: 64 → 128
2. 调整特征提取：
   - 增加扩张率范围
   - 使用双向S4

**如果基线过拟合（训练好但测试差）**：
1. 减少模型容量
2. 增加正则化：
   - `s4_dropout`: 0.0 → 0.1
   - 使用更强的数据增强

**如果训练不稳定**：
1. 减小学习率
2. 使用梯度裁剪（已在代码中实现）
3. 减小批次大小

---

## 代码结构

```
SSSD/
├── src/
│   ├── imputers/
│   │   ├── ImplicitExplicitDiffusion.py   # 🔥 隐式-显式扩散模型（核心）
│   │   ├── S4Model.py                      # S4状态空间模型实现
│   │   ├── SSSDS4Imputer.py               # 原SSSD模型（参考）
│   │   └── DiffWaveImputer.py             # 原DiffWave模型（参考）
│   │
│   ├── utils/
│   │   └── util.py                         # 工具函数（扩散过程等）
│   │
│   ├── config/
│   │   └── config_ImplicitExplicit_LD2011.json  # 配置文件
│   │
│   ├── data_loader_ld2011.py               # 🔥 数据加载和预处理
│   ├── train_ld2011.py                     # 🔥 训练脚本
│   ├── evaluate_ld2011.py                  # 🔥 评估脚本
│   │
│   └── requirements.txt                    # Python依赖
│
├── run_experiments.sh                      # 🔥 自动化实验脚本
├── README_ImplicitExplicitDiffusion.md     # 🔥 本文档
│
├── datasets/                               # 预处理后的数据
│   ├── train_ld2011.npy
│   └── test_ld2011.npy
│
└── results/                                # 实验结果
    └── ld2011/
        ├── T200_beta00.0001_betaT0.02_missing20/
        │   ├── *.pkl                       # 模型检查点
        │   ├── training_loss_curve.png     # 训练曲线
        │   ├── losses/                     # 损失历史
        │   └── visualizations/             # 可视化结果
        ├── T200_beta00.0001_betaT0.02_missing30/
        ├── ...
        └── evaluation_results.csv          # 评估结果表格
```

### 核心文件详解

#### 1. ImplicitExplicitDiffusion.py

**主要类和方法**：

```python
# 隐式特征提取
class ImplicitFeatureExtractor(nn.Module):
    def __init__(self, channels, dilation_rates, kernel_size)
    def forward(self, x)  # 输入: [B,C,L] -> 输出: [B,C,L]

# 显式特征提取
class ExplicitFeatureExtractor(nn.Module):
    def __init__(self, channels, s4_lmax, s4_d_state, ...)
    def forward(self, x)  # 输入: [B,C,L] -> 输出: [B,C,L]

# 残差块
class ImplicitExplicitResidualBlock(nn.Module):
    def forward(self, (x, cond, diffusion_step_embed))
    # 返回: (残差输出, 跳跃连接)

# 主模型
class ImplicitExplicitDiffusion(nn.Module):
    def forward(self, (noise, conditional, mask, diffusion_steps))
    # 返回: 噪声预测
```

**关键流程**（第690-740行）：
```python
def forward(self, input_data):
    noise, conditional, mask, diffusion_steps = input_data

    # 1. 准备条件信息（观测+掩码）
    conditional = conditional * mask
    conditional = torch.cat([conditional, mask.float()], dim=1)

    # 2. 初始特征提取
    x = self.init_conv(noise)

    # 3. 隐式-显式特征提取（残差块组）
    x = self.residual_layer((x, conditional, diffusion_steps))

    # 4. 最终输出
    y = self.final_conv(x)
    return y
```

#### 2. train_ld2011.py

**主要函数**：

```python
def create_random_mask(data, missing_ratio):
    """创建随机缺失掩码"""

def train(output_directory, ckpt_iter, n_iters, ...):
    """主训练函数"""
    # 1. 设置实验路径
    # 2. 创建模型和优化器
    # 3. 加载检查点（如果有）
    # 4. 加载数据
    # 5. 训练循环
    # 6. 绘制损失曲线
```

**训练循环**（第177-269行）：
```python
for batch_data in train_loader:
    batch = batch_data[0].to(device)

    # 创建掩码
    mask = create_random_mask(batch, missing_ratio)

    # 前向传播
    X = (batch, batch, mask, loss_mask)
    loss = training_loss(net, nn.MSELoss(), X, ...)

    # 反向传播
    loss.backward()
    optimizer.step()
```

#### 3. evaluate_ld2011.py

**主要函数**：

```python
def calculate_metrics(predictions, ground_truth, mask):
    """计算MAE和RMSE"""

def evaluate(ckpt_path, missing_ratio, n_samples, device):
    """评估单个缺失率"""
    # 1. 加载模型
    # 2. 加载测试数据
    # 3. 执行采样（扩散逆过程）
    # 4. 计算指标
    # 5. 可视化结果

def evaluate_multiple_missing_ratios(...):
    """评估多个缺失率并生成报告"""
```

#### 4. data_loader_ld2011.py

**主要类**：

```python
class LD2011Dataset:
    def _load_data(self)          # 加载原始txt文件
    def _preprocess(self)          # 数据清洗和归一化
    def _split_train_test(self)    # 分割训练/测试集
    def _create_sequences(self)    # 创建滑动窗口序列
    def inverse_transform(self)    # 反归一化

def prepare_data_for_training(...):  # 完整的数据准备流程
```

---

## 实验结果

### 目标指标（论文）

| 缺失率 | MAE  | RMSE |
|--------|------|------|
| 20%    | 0.272 | 0.389 |
| 30%    | 0.297 | 0.424 |
| 40%    | 0.334 | 0.477 |
| 50%    | 0.378 | 0.540 |
| 60%    | 0.450 | 0.655 |
| 70%    | 0.541 | 0.776 |
| 80%    | 0.732 | 1.049 |

### 查看你的实验结果

训练完成后，结果保存在：
- CSV表格：`results/ld2011/evaluation_results.csv`
- 可视化图表：`results/ld2011/metrics_vs_missing_ratio.png`
- 各缺失率详细结果：`results/ld2011/T*_missing*/visualizations/`

### 结果分析

**如何判断模型性能**：

1. **对比目标指标**：
   - MAE和RMSE应该接近或低于目标值
   - 误差随缺失率增加而增大（这是正常的）

2. **观察训练曲线**：
   - 损失应该平稳下降
   - 没有剧烈波动或发散
   - 最终损失应该收敛

3. **检查可视化结果**：
   - 预测值应该接近真实值
   - 尤其注意缺失位置的预测质量
   - 应该能捕获数据的时序模式

**如果性能不佳**：

1. **检查数据**：
   - 数据是否正确加载
   - 归一化是否正常
   - 掩码是否正确生成

2. **检查训练**：
   - 训练是否收敛
   - 学习率是否合适
   - 是否需要更多迭代

3. **调整模型**：
   - 参考超参数调优部分
   - 尝试不同的架构配置

---

## 常见问题

### Q1: 训练时显存不足怎么办？

**解决方案**：

1. 减小批次大小：
```json
"batch_size": 8  // 或更小
```

2. 减小模型大小：
```json
"res_channels": 128,
"num_res_layers": 20
```

3. 使用梯度累积：
```python
# 在train_ld2011.py中修改
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q2: 训练太慢怎么办？

**解决方案**：

1. 使用混合精度训练：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = training_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

2. 减少扩散步数（推理阶段）：
```json
"T": 100  // 从200减少到100
```

3. 使用更快的GPU（A40而不是A10）

### Q3: 如何在不同GPU上训练？

**方案1：指定GPU设备**
```bash
# 使用A40 (cuda:0)
python train_ld2011.py --device cuda:0

# 使用A10 (cuda:1)
python train_ld2011.py --device cuda:1
```

**方案2：使用环境变量**
```bash
CUDA_VISIBLE_DEVICES=0 python train_ld2011.py  # 使用GPU 0
CUDA_VISIBLE_DEVICES=1 python train_ld2011.py  # 使用GPU 1
```

### Q4: 如何继续中断的训练？

训练脚本会自动保存检查点，继续训练：

```bash
# 默认会自动从最新检查点恢复
python train_ld2011.py --config config/...json
```

或手动指定：
```json
"ckpt_iter": 10000  // 从第10000次迭代恢复
```

### Q5: 数据加载失败怎么办？

**常见原因**：

1. 文件路径错误：
```python
# 检查路径是否正确
data_path = "/home/zhu/sssdtcn/LD2011_2014.txt"
```

2. 文件格式问题：
   - 检查分隔符（分号、制表符、逗号）
   - 检查是否有表头
   - 检查数值格式（欧洲格式用逗号作小数点）

3. 编码问题：
```python
# 在data_loader_ld2011.py中尝试不同编码
pd.read_csv(path, encoding='utf-8')  # 或 'latin1', 'gbk'
```

### Q6: 如何调整到最佳性能？

**系统化调优流程**：

**第1步：建立基线**
- 使用默认参数训练
- 记录MAE和RMSE

**第2步：数据层面**
- 尝试不同的window_size（50, 100, 150）
- 尝试不同的stride（window_size // 2）
- 检查数据归一化效果

**第3步：训练层面**
- 调整学习率（1e-4, 2e-4, 5e-4）
- 增加训练迭代次数
- 尝试学习率调度

**第4步：模型层面**
- 调整隐式模块扩张率
- 调整S4状态维度
- 调整模型容量（res_channels, num_res_layers）

**第5步：扩散层面**
- 调整扩散步数T
- 调整beta_0和beta_T

### Q7: 模型预测效果不好怎么办？

**诊断步骤**：

1. **检查训练曲线**：
   ```
   results/ld2011/T*_missing*/training_loss_curve.png
   ```
   - 损失是否收敛？
   - 是否有异常波动？

2. **检查可视化结果**：
   ```
   results/ld2011/T*_missing*/visualizations/
   ```
   - 预测是否跟随真实值的趋势？
   - 是否只是输出常数？

3. **检查数据处理**：
   - 打印几个样本检查数据格式
   - 确认掩码生成正确

4. **调整超参数**：
   - 参考超参数调优部分
   - 尝试不同配置

### Q8: 如何解读训练日志？

典型的训练日志：
```
迭代 100/50000 | Epoch 1 | 损失: 0.523418
迭代 200/50000 | Epoch 1 | 损失: 0.412356
...
```

**正常情况**：
- 损失逐渐下降
- 最终稳定在较低值（< 0.1）

**异常情况**：
- 损失NaN或Inf → 学习率太大或数值不稳定
- 损失不下降 → 学习率太小或模型有问题
- 损失剧烈波动 → 批次大小太小或数据有问题

### Q9: 如何可视化自己的数据？

```python
import matplotlib.pyplot as plt
import numpy as np

# 加载预处理数据
data = np.load('datasets/train_ld2011.npy')

# 绘制一个样本
sample = data[0]  # 形状: [length, features]

plt.figure(figsize=(15, 8))
for i in range(min(5, sample.shape[1])):  # 绘制前5个特征
    plt.subplot(5, 1, i+1)
    plt.plot(sample[:, i])
    plt.ylabel(f'Feature {i+1}')
    plt.grid(True)
plt.xlabel('Time Step')
plt.tight_layout()
plt.savefig('data_sample.png')
```

### Q10: 如何导出模型用于部署？

```python
import torch
from imputers.ImplicitExplicitDiffusion import ImplicitExplicitDiffusion

# 加载模型
model = ImplicitExplicitDiffusion(**model_config)
checkpoint = torch.load('path/to/checkpoint.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 导出为TorchScript
example_input = (
    torch.randn(1, 14, 100),  # noise
    torch.randn(1, 14, 100),  # conditional
    torch.ones(1, 14, 100),   # mask
    torch.tensor([[50]])      # diffusion_step
)

traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_deployed.pt')
```

---

## 参考文献

1. **SSSD论文**: "Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models"
   - 链接: https://openreview.net/forum?id=hHiIbk7ApW

2. **S4论文**: "Efficiently Modeling Long Sequences with Structured State Spaces"
   - 链接: https://arxiv.org/abs/2111.00396

3. **DDPM论文**: "Denoising Diffusion Probabilistic Models"
   - 链接: https://arxiv.org/abs/2006.11239

4. **WaveNet论文**: "WaveNet: A Generative Model for Raw Audio"
   - 链接: https://arxiv.org/abs/1609.03499

---

## 致谢

本项目基于以下开源项目：
- [SSSD](https://github.com/AI4HealthUOL/SSSD) - 基础架构
- [S4](https://github.com/HazyResearch/state-spaces) - S4模型实现
- [DiffWave](https://github.com/philsyn/DiffWave-Vocoder) - WaveNet架构

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**祝实验顺利！** 🚀
