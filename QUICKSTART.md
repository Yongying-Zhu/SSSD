# 快速开始指南

## 🚀 5分钟快速上手

### 步骤 0: 环境准备

**重要提示**：请在你的服务器上执行以下步骤（zhu@lys-PowerEdge-R750xa）

#### 检查Python和CUDA
```bash
python --version  # 应该 >= 3.8
nvidia-smi       # 检查GPU状态
```

#### 安装依赖

**方案1：使用conda（推荐）**
```bash
# 创建新环境
conda create -n implicit_explicit python=3.8
conda activate implicit_explicit

# 安装PyTorch（根据你的CUDA版本选择）
# 对于CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装其他依赖
cd /home/user/SSSD/src
pip install numpy pandas matplotlib scikit-learn tqdm scipy opt_einsum einops
```

**方案2：使用pip**
```bash
cd /home/user/SSSD/src
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install numpy pandas matplotlib scikit-learn tqdm scipy opt_einsum einops
```

#### 验证安装
```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

### 步骤 1: 准备数据

**1.1 确认数据文件位置**

请确认你的LD2011_2014.txt数据文件路径，例如：
```bash
ls -lh /home/zhu/sssdtcn/LD2011_2014.txt
```

如果路径不同，需要修改以下文件：
- `src/data_loader_ld2011.py` (最后几行的 `__main__` 部分)
- `run_experiments.sh` (第19行的 `DATA_PATH`)

**1.2 运行数据预处理**

```bash
cd /home/user/SSSD/src

# 修改data_loader_ld2011.py中的数据路径
# 找到文件末尾的这一行：
# data_path = "/home/zhu/sssdtcn/LD2011_2014.txt"
# 修改为你的实际路径

python data_loader_ld2011.py
```

**预期输出**：
```
正在从 ... 加载数据...
原始数据形状: (时间步数, 特征数)
预处理后数据形状: (时间步数, 特征数)
训练集形状: (样本数, 序列长度, 特征数)
测试集形状: (样本数, 序列长度, 特征数)
训练序列数: XXX
测试序列数: XXX

数据预处理完成！
训练数据保存至: /home/user/SSSD/datasets/train_ld2011.npy
测试数据保存至: /home/user/SSSD/datasets/test_ld2011.npy
```

**如果出错**：
- 检查数据文件路径是否正确
- 检查文件格式（分隔符、编码等）
- 查看错误信息并根据提示修改

### 步骤 2: 训练模型（单个缺失率）

**2.1 快速测试（5000次迭代，约5-10分钟）**

```bash
cd /home/user/SSSD/src

# 修改配置文件，设置较少的迭代次数用于测试
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

**预期输出**：
```
================================================================================
隐式-显式扩散模型 - 训练脚本
================================================================================
...
使用设备: cuda:0
创建隐式-显式扩散模型...
ImplicitExplicitDiffusion Parameters: X.XXX M

训练进度:   0%|          | 0/5000
迭代 100/5000 | Epoch 1 | 损失: 0.XXXX
迭代 200/5000 | Epoch 1 | 损失: 0.XXXX
...
训练完成！
损失曲线已保存: ./results/ld2011/.../training_loss_curve.png
```

**2.2 完整训练（50000次迭代，约2-4小时）**

如果快速测试成功，可以开始完整训练：

```bash
# 缺失率20%
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0

# 缺失率50%
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.5 \
    --device cuda:0
```

### 步骤 3: 评估模型

**3.1 评估单个缺失率**

```bash
cd /home/user/SSSD/src

python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

**预期输出**：
```
============================================================
评估结果 (缺失率: 20%)
============================================================
MAE:  0.XXXX
RMSE: 0.XXXX
============================================================

可视化结果已保存到: ./results/ld2011/.../visualizations/
```

**3.2 评估所有缺失率**

前提：已经训练了多个缺失率的模型

```bash
python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --eval_all \
    --device cuda:0
```

**预期输出**：
```
============================================================
评估结果汇总
============================================================
Missing Ratio    MAE    RMSE
          20%  0.XXX   0.XXX
          30%  0.XXX   0.XXX
          ...
============================================================

评估结果已保存到: ./results/ld2011/evaluation_results.csv
指标对比图已保存到: ./results/ld2011/metrics_vs_missing_ratio.png
```

### 步骤 4: 查看结果

**4.1 训练损失曲线**

```bash
# 查看某个缺失率的训练曲线
xdg-open ./results/ld2011/T200_beta00.0001_betaT0.02_missing20/training_loss_curve.png
```

**4.2 可视化插补结果**

```bash
# 查看插补效果
ls ./results/ld2011/T200_beta00.0001_betaT0.02_missing20/visualizations/
```

**4.3 评估指标表格**

```bash
# CSV格式
cat ./results/ld2011/evaluation_results.csv

# 或者用Excel打开
```

---

## 🔧 常见问题快速解决

### 问题1: ModuleNotFoundError: No module named 'torch'

**解决**：
```bash
pip install torch torchvision torchaudio
```

### 问题2: CUDA out of memory

**解决**：减小批次大小

编辑 `config/config_ImplicitExplicit_LD2011.json`：
```json
"batch_size": 8  // 从16改为8
```

或者减小模型大小：
```json
"res_channels": 128,  // 从256改为128
"num_res_layers": 20  // 从36改为20
```

### 问题3: 数据文件找不到

**解决**：

1. 确认数据文件实际路径：
```bash
find /home -name "LD2011_2014.txt" 2>/dev/null
```

2. 修改 `src/data_loader_ld2011.py`，在 `__main__` 部分：
```python
data_path = "/你的实际路径/LD2011_2014.txt"
```

3. 修改 `run_experiments.sh`：
```bash
DATA_PATH="/你的实际路径/LD2011_2014.txt"
```

### 问题4: 训练中断了怎么办？

**解决**：重新运行相同的命令，会自动从最新检查点恢复

```bash
# 直接重新运行，会自动恢复
python train_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --missing_ratio 0.2 \
    --device cuda:0
```

### 问题5: 如何使用A10而不是A40？

**解决**：修改设备参数

```bash
# 使用A10 (cuda:1)
python train_ld2011.py ... --device cuda:1

# 或者修改配置文件
"device": "cuda:1"
```

---

## 📊 完整实验流程（训练所有缺失率）

### 方案1：使用自动化脚本

```bash
cd /home/user/SSSD

# 1. 修改脚本中的数据路径
nano run_experiments.sh
# 找到第19行，修改为你的实际数据路径

# 2. 运行脚本
./run_experiments.sh
```

脚本会自动：
1. 检查环境和数据
2. 预处理数据
3. 训练所有缺失率（20%-80%）
4. 评估并生成报告

### 方案2：手动运行

```bash
cd /home/user/SSSD/src

# 1. 数据预处理
python data_loader_ld2011.py

# 2. 训练所有缺失率
for ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    echo "训练缺失率: ${ratio}"
    python train_ld2011.py \
        --config config/config_ImplicitExplicit_LD2011.json \
        --missing_ratio ${ratio} \
        --device cuda:0
done

# 3. 评估所有缺失率
python evaluate_ld2011.py \
    --config config/config_ImplicitExplicit_LD2011.json \
    --eval_all \
    --device cuda:0
```

---

## 🎯 预期时间

基于默认配置（50000次迭代）：

| 任务 | 时间（A40） | 时间（A10） |
|------|------------|------------|
| 数据预处理 | <1分钟 | <1分钟 |
| 单次训练 | 2-3小时 | 3-4小时 |
| 单次评估 | 5-10分钟 | 10-15分钟 |
| 完整实验（7个缺失率） | 14-21小时 | 21-28小时 |

**建议**：
- 先用5000次迭代快速测试（约10分钟）
- 确认无误后再运行完整实验
- 可以使用screen或tmux在后台运行

---

## 📝 检查清单

完成快速开始前，确认：

- [ ] Python >= 3.8 已安装
- [ ] PyTorch已安装且CUDA可用
- [ ] 所有依赖包已安装
- [ ] LD2011_2014.txt数据文件路径已确认
- [ ] 数据预处理成功运行
- [ ] 至少一个缺失率的训练成功
- [ ] 评估脚本能够正常运行
- [ ] 可以查看到结果文件

---

## 🆘 获取帮助

如果遇到问题：

1. 查看 `README_ImplicitExplicitDiffusion.md` 的"常见问题"部分
2. 检查错误信息并Google搜索
3. 查看代码注释理解每个部分的功能
4. 在服务器上直接运行调试

---

**祝实验顺利！** 🎉
