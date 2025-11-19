#!/bin/bash

################################################################################
# 项目配置脚本
# 用于设置数据路径和环境配置
################################################################################

echo "================================================================================"
echo "隐式-显式扩散模型 - 项目配置向导"
echo "================================================================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 步骤1: 激活虚拟环境提示
echo ""
echo -e "${YELLOW}步骤 1: 确认虚拟环境${NC}"
echo "请确保已激活 AnYujin 虚拟环境"
echo ""
echo "如果还未激活，请运行："
echo "  conda activate AnYujin"
echo ""
read -p "已激活虚拟环境? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}请先激活虚拟环境后重新运行此脚本${NC}"
    exit 1
fi

# 步骤2: 安装缺失依赖
echo ""
echo -e "${YELLOW}步骤 2: 检查并安装缺失的依赖${NC}"
echo "检查 opt_einsum 是否已安装..."

if python -c "import opt_einsum" 2>/dev/null; then
    echo -e "${GREEN}✓ opt_einsum 已安装${NC}"
else
    echo -e "${YELLOW}正在安装 opt_einsum...${NC}"
    pip install opt_einsum
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ opt_einsum 安装成功${NC}"
    else
        echo -e "${RED}✗ opt_einsum 安装失败${NC}"
        exit 1
    fi
fi

# 步骤3: 运行环境验证
echo ""
echo -e "${YELLOW}步骤 3: 运行环境验证${NC}"
cd /home/user/SSSD
python check_environment.py

if [ $? -ne 0 ]; then
    echo -e "${RED}环境验证失败，请检查错误信息${NC}"
    exit 1
fi

# 步骤4: 配置数据路径
echo ""
echo -e "${YELLOW}步骤 4: 配置数据文件路径${NC}"
echo ""
echo "请输入 LD2011_2014.txt 数据文件的完整路径"
echo "示例: /home/zhu/sssdtcn/LD2011_2014.txt"
echo ""
read -p "数据文件路径: " DATA_PATH

# 验证文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${YELLOW}⚠ 警告: 文件不存在: $DATA_PATH${NC}"
    read -p "是否继续配置? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "配置已取消"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 文件存在: $DATA_PATH${NC}"
fi

# 更新data_loader_ld2011.py中的路径
echo ""
echo "正在更新 data_loader_ld2011.py 中的数据路径..."

# 创建备份
cp src/data_loader_ld2011.py src/data_loader_ld2011.py.backup

# 更新路径
sed -i "s|data_path = \".*\"|data_path = \"$DATA_PATH\"|g" src/data_loader_ld2011.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ data_loader_ld2011.py 已更新${NC}"
else
    echo -e "${RED}✗ 更新失败${NC}"
    exit 1
fi

# 更新run_experiments.sh中的路径
echo "正在更新 run_experiments.sh 中的数据路径..."

# 创建备份
cp run_experiments.sh run_experiments.sh.backup

# 更新路径
sed -i "s|DATA_PATH=\".*\"|DATA_PATH=\"$DATA_PATH\"|g" run_experiments.sh

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ run_experiments.sh 已更新${NC}"
else
    echo -e "${RED}✗ 更新失败${NC}"
    exit 1
fi

# 步骤5: 选择GPU设备
echo ""
echo -e "${YELLOW}步骤 5: 选择GPU设备${NC}"
echo ""
echo "您的服务器有两张GPU:"
echo "  1. cuda:0 - A40 (推荐，速度更快)"
echo "  2. cuda:1 - A10"
echo ""
read -p "请选择GPU设备 (0/1, 默认0): " GPU_CHOICE

if [ -z "$GPU_CHOICE" ]; then
    GPU_CHOICE=0
fi

GPU_DEVICE="cuda:$GPU_CHOICE"
echo -e "${GREEN}✓ 已选择设备: $GPU_DEVICE${NC}"

# 更新配置文件
echo "正在更新配置文件中的GPU设置..."
sed -i "s|\"device\": \"cuda:.*\"|\"device\": \"$GPU_DEVICE\"|g" src/config/config_ImplicitExplicit_LD2011.json

# 步骤6: 创建必要目录
echo ""
echo -e "${YELLOW}步骤 6: 创建必要目录${NC}"

mkdir -p datasets
mkdir -p results/ld2011

echo -e "${GREEN}✓ 目录创建完成${NC}"

# 配置完成
echo ""
echo "================================================================================"
echo -e "${GREEN}✓ 配置完成！${NC}"
echo "================================================================================"
echo ""
echo "配置摘要:"
echo "  - 虚拟环境: AnYujin"
echo "  - 数据文件: $DATA_PATH"
echo "  - GPU设备: $GPU_DEVICE"
echo ""
echo "下一步操作:"
echo ""
echo "1. 【快速测试】运行5000次迭代测试（约10分钟）："
echo "   cd /home/user/SSSD/src"
echo "   python data_loader_ld2011.py  # 预处理数据"
echo ""
echo "   # 修改配置文件，设置较少迭代次数进行测试"
echo "   # 编辑 config/config_ImplicitExplicit_LD2011.json"
echo "   # 将 \"n_iters\": 50000 改为 \"n_iters\": 5000"
echo ""
echo "   python train_ld2011.py --missing_ratio 0.2"
echo ""
echo "2. 【完整训练】训练所有缺失率（需要14-21小时）："
echo "   cd /home/user/SSSD"
echo "   ./run_experiments.sh"
echo ""
echo "3. 【单个缺失率】训练单个缺失率："
echo "   cd /home/user/SSSD/src"
echo "   python train_ld2011.py --missing_ratio 0.2 --device $GPU_DEVICE"
echo ""
echo "================================================================================"

# 询问是否立即运行环境验证
echo ""
read -p "是否查看详细的环境信息? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "CUDA设备信息:"
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"
fi

echo ""
echo -e "${GREEN}配置完成！祝实验顺利！${NC}"
