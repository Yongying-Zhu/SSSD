# 完整安装和使用指南

## 🎯 您当前的状态

✅ **虚拟环境**: AnYujin (已创建)
✅ **PyTorch**: 2.2.0+cu118 (已安装，支持CUDA 11.8)
✅ **大部分依赖**: 已安装
⚠️ **需要安装**: opt_einsum
⏳ **需要配置**: 数据文件路径

---

## 📋 快速开始（3步走）

### 第1步：完成环境配置（5分钟）

在您的服务器上运行：

```bash
# 1. 激活虚拟环境
conda activate AnYujin

# 2. 进入项目目录
cd /home/user/SSSD

# 3. 运行配置脚本（会自动安装缺失的包并配置路径）
./setup_config.sh
```

**配置脚本会自动完成**：
- ✅ 检查并安装 opt_einsum
- ✅ 验证所有依赖
- ✅ 配置数据文件路径
- ✅ 选择GPU设备
- ✅ 创建必要目录

### 第2步：快速测试（10-15分钟）

```bash
# 1. 预处理数据
cd /home/user/SSSD/src
python data_loader_ld2011.py

# 2. 快速训练测试（5000次迭代，约10分钟）
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011_QuickTest.json \
    --missing_ratio 0.2 \
    --device cuda:0

# 3. 评估测试结果
python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011_QuickTest.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

### 第3步：完整实验（可选，14-21小时）

如果快速测试成功，可以运行完整实验：

```bash
cd /home/user/SSSD
./run_experiments.sh
```

---

## 🔧 手动配置（如果自动配置脚本失败）

### 1. 安装缺失的包

```bash
conda activate AnYujin
pip install opt_einsum
```

### 2. 验证环境

```bash
cd /home/user/SSSD
python check_environment.py
```

预期输出：
```
✅ 所有依赖检查通过！环境配置正确。
✅ CUDA可用
   GPU 0: NVIDIA A40
   GPU 1: NVIDIA A10
✅ PyTorch环境测试通过！
```

### 3. 配置数据路径

**步骤 3.1**: 找到您的数据文件

```bash
# 在服务器上查找数据文件
find /home -name "LD2011_2014.txt" 2>/dev/null
```

假设找到的路径是：`/home/zhu/sssdtcn/LD2011_2014.txt`

**步骤 3.2**: 修改 `src/data_loader_ld2011.py`

用文本编辑器打开文件，找到最后的 `if __name__ == "__main__":` 部分，修改：

```python
# 找到这一行（大约在第240行）
data_path = "/home/zhu/sssdtcn/LD2011_2014.txt"

# 修改为您的实际路径
data_path = "/您的实际路径/LD2011_2014.txt"
```

**步骤 3.3**: 修改 `run_experiments.sh`

找到第19行，修改：

```bash
DATA_PATH="/home/zhu/sssdtcn/LD2011_2014.txt"

# 修改为您的实际路径
DATA_PATH="/您的实际路径/LD2011_2014.txt"
```

### 4. 选择GPU设备

**步骤 4.1**: 检查GPU状态

```bash
nvidia-smi
```

**步骤 4.2**: 修改配置文件

编辑 `src/config/config_ImplicitExplicit_LD2011.json`：

```json
{
    "train_config": {
        ...
        "device": "cuda:0"  // cuda:0=A40, cuda:1=A10
    }
}
```

---

## 📝 详细使用说明

### 数据预处理

```bash
cd /home/user/SSSD/src
python data_loader_ld2011.py
```

**预期输出**：
```
正在从 /path/to/LD2011_2014.txt 加载数据...
原始数据形状: (时间步数, 特征数)
预处理后数据形状: (时间步数, 特征数)
训练集形状: (样本数, 100, 14)
测试集形状: (样本数, 100, 14)

数据预处理完成！
训练数据保存至: ./datasets/train_ld2011.npy
测试数据保存至: ./datasets/test_ld2011.npy
```

### 模型训练

#### 方案1：快速测试（推荐首次使用）

使用快速测试配置（5000次迭代，小模型）：

```bash
cd /home/user/SSSD/src

python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011_QuickTest.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

**配置说明**：
- 迭代次数：5000（而不是50000）
- 模型大小：更小（res_channels=128, num_res_layers=20）
- 批次大小：8
- 时间：约10-15分钟（A40）

#### 方案2：完整训练

使用完整配置（50000次迭代，大模型）：

```bash
cd /home/user/SSSD/src

python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

**配置说明**：
- 迭代次数：50000
- 模型大小：完整（res_channels=256, num_res_layers=36）
- 批次大小：16
- 时间：约2-3小时（A40）

#### 方案3：批量训练所有缺失率

```bash
cd /home/user/SSSD
./run_experiments.sh
```

这会自动训练20%-80%所有缺失率（需要14-21小时）

### 模型评估

```bash
cd /home/user/SSSD/src

# 评估单个缺失率
python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0

# 评估所有缺失率（需要先训练完所有缺失率）
python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --eval_all \
    --device cuda:0
```

---

## 🔍 检查结果

### 训练结果

**位置**：`results/ld2011/T200_beta00.0001_betaT0.02_missing{X}/`

**包含文件**：
- `*.pkl`: 模型检查点
- `training_loss_curve.png`: 训练损失曲线
- `losses/`: 损失历史数据

**如何查看**：
```bash
# 查看损失曲线
ls results/ld2011/T200_beta00.0001_betaT0.02_missing20/training_loss_curve.png
```

### 评估结果

**位置**：
- 可视化：`results/ld2011/T200_beta00.0001_betaT0.02_missing{X}/visualizations/`
- 汇总表：`results/ld2011/evaluation_results.csv`

**如何查看**：
```bash
# 查看评估报告
cat results/ld2011/evaluation_results.csv

# 查看可视化结果
ls results/ld2011/T200_beta00.0001_betaT0.02_missing20/visualizations/
```

---

## 💡 常见问题

### Q1: 如何确认我在正确的虚拟环境中？

```bash
# 检查当前环境
conda info --envs

# 应该看到 AnYujin 前面有 * 号
# * AnYujin    /path/to/AnYujin
```

### Q2: 如何在后台运行长时间训练？

使用 `screen` 或 `tmux`：

```bash
# 方案1：使用screen
screen -S implicit_explicit
conda activate AnYujin
cd /home/user/SSSD
./run_experiments.sh

# 按 Ctrl+A 然后按 D 断开（训练继续进行）
# 重新连接：screen -r implicit_explicit

# 方案2：使用nohup
nohup ./run_experiments.sh > experiment.log 2>&1 &
```

### Q3: 训练过程中显存不足怎么办？

**解决方案1**：使用快速测试配置（已经是小模型）

**解决方案2**：进一步减小批次大小

编辑配置文件，修改：
```json
"batch_size": 4  // 从8改为4
```

**解决方案3**：使用A40而不是A10

```bash
--device cuda:0  // A40有更多显存
```

### Q4: 如何查看训练进度？

```bash
# 方案1：直接观察输出
# 会显示：迭代 100/5000 | Epoch 1 | 损失: 0.XXXX

# 方案2：查看损失历史
cd results/ld2011/T200_beta00.0001_betaT0.02_missing20/losses/
python -c "import numpy as np; loss = np.load('loss_history.npy'); print(f'最新损失: {loss[-1]:.6f}')"
```

### Q5: 数据预处理失败怎么办？

**常见原因和解决方案**：

1. **文件路径错误**：
   ```bash
   # 确认文件存在
   ls -lh /home/zhu/sssdtcn/LD2011_2014.txt
   ```

2. **文件格式问题**：
   ```bash
   # 查看文件前几行
   head -5 /home/zhu/sssdtcn/LD2011_2014.txt
   ```

3. **权限问题**：
   ```bash
   # 检查文件权限
   ls -l /home/zhu/sssdtcn/LD2011_2014.txt
   ```

### Q6: 如何中断并恢复训练？

**中断训练**：
- 按 `Ctrl+C`

**恢复训练**：
```bash
# 使用相同的命令重新运行，会自动从最新检查点恢复
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

训练脚本会自动：
1. 查找最新的检查点（`ckpt_iter: "max"`）
2. 加载模型和优化器状态
3. 从上次中断的迭代继续

---

## 📊 预期性能

### 快速测试（5000次迭代）

**目的**：验证代码可以运行，模型可以学习

**预期结果**：
- 训练损失下降到 0.2-0.5
- MAE: 约 0.5-0.8
- RMSE: 约 0.7-1.2

⚠️ **注意**：快速测试的性能不会很好，这是正常的

### 完整训练（50000次迭代）

**目标性能**（20%缺失率）：
- MAE: 0.272
- RMSE: 0.389

**实际性能可能**：
- MAE: 0.25-0.35
- RMSE: 0.35-0.45

如果第一次训练没有达到目标，可以：
1. 增加训练迭代次数（100000）
2. 调整学习率
3. 调整模型大小
4. 参考 README 中的超参数调优指南

---

## 🎯 推荐的实验流程

### 第1天：环境配置和快速测试

**上午**（1小时）：
1. 运行 `setup_config.sh` 配置环境
2. 运行 `check_environment.py` 验证环境
3. 运行 `data_loader_ld2011.py` 预处理数据

**下午**（2小时）：
1. 运行快速测试（5000次迭代）
2. 检查训练是否正常
3. 查看损失曲线和初步结果

### 第2-3天：完整实验

**启动完整训练**（后台运行）：
```bash
screen -S full_training
conda activate AnYujin
cd /home/user/SSSD
./run_experiments.sh
# Ctrl+A, D (断开但保持运行)
```

**期间可以**：
- 定期检查进度
- 查看训练日志
- 监控GPU使用情况

### 第4天：结果分析

1. 查看所有训练损失曲线
2. 运行评估脚本
3. 查看可视化结果
4. 分析性能指标
5. 如需调优，参考README

---

## 📚 相关文档

- **快速开始**: `QUICKSTART.md`
- **详细说明**: `README_ImplicitExplicitDiffusion.md`
- **项目总结**: `PROJECT_SUMMARY.md`

---

## 🆘 获取帮助

如果遇到问题：

1. **首先检查**：
   - 虚拟环境是否激活
   - 数据文件路径是否正确
   - GPU是否可用（`nvidia-smi`）

2. **查看错误信息**：
   - 仔细阅读错误提示
   - 根据错误类型查找对应FAQ

3. **查看文档**：
   - 本文档的常见问题部分
   - README的FAQ部分
   - 代码注释

4. **调试技巧**：
   ```bash
   # 启用Python调试模式
   python -u train_ld2011.py ... 2>&1 | tee train.log
   ```

---

## ✅ 配置检查清单

使用前请确认：

- [ ] 虚拟环境 AnYujin 已激活
- [ ] opt_einsum 已安装
- [ ] 运行过 `check_environment.py` 且全部通过
- [ ] LD2011_2014.txt 数据文件路径已确认
- [ ] 数据路径已在代码中配置
- [ ] GPU设备已选择
- [ ] 数据预处理成功完成
- [ ] datasets/ 目录下有 train_ld2011.npy 和 test_ld2011.npy

完成以上检查后，即可开始训练！

---

**祝实验顺利！** 🚀
