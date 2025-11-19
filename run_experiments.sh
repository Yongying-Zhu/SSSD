#!/bin/bash

################################################################################
# 隐式-显式扩散模型实验运行脚本
#
# 该脚本用于运行完整的实验流程：
# 1. 数据预处理
# 2. 在不同缺失率下训练模型 (20%-80%)
# 3. 评估模型性能并生成报告
#
# 作者: Claude AI
# 日期: 2025-11-19
################################################################################

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 设置路径
DATA_PATH="/home/zhu/sssdtcn/LD2011_2014.txt"
SRC_DIR="/home/user/SSSD/src"
CONFIG_FILE="${SRC_DIR}/config/config_ImplicitExplicit_LD2011.json"

# GPU设置 (根据你的服务器配置选择)
# cuda:0 -> A40
# cuda:1 -> A10
GPU_DEVICE="cuda:0"

# 训练参数
# 迭代次数：可以根据需要调整
# - 快速测试: 5000
# - 正常训练: 30000-50000
# - 完整训练: 100000+
N_ITERS=50000

# 缺失率列表（20%-80%，步长10%）
MISSING_RATIOS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)

################################################################################
# 函数定义
################################################################################

print_header() {
    echo -e "${BLUE}"
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "文件不存在: $1"
        return 1
    fi
    return 0
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "未检测到NVIDIA GPU"
        return 1
    fi

    print_info "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    return 0
}

################################################################################
# 主流程
################################################################################

# 切换到源代码目录
cd ${SRC_DIR}

print_header "隐式-显式扩散模型 - 实验运行脚本"

# 检查GPU
print_info "检查GPU状态..."
if ! check_gpu; then
    print_error "GPU检查失败，退出"
    exit 1
fi
print_success "GPU检查通过"

# 检查数据文件
print_info "检查数据文件..."
if ! check_file "${DATA_PATH}"; then
    print_error "数据文件不存在，请检查路径: ${DATA_PATH}"
    exit 1
fi
print_success "数据文件检查通过"

################################################################################
# 步骤 1: 数据预处理
################################################################################

print_header "步骤 1: 数据预处理"

if [ -f "./datasets/train_ld2011.npy" ] && [ -f "./datasets/test_ld2011.npy" ]; then
    print_info "检测到已存在的预处理数据"
    read -p "是否重新预处理? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python data_loader_ld2011.py
    else
        print_info "跳过数据预处理"
    fi
else
    print_info "开始数据预处理..."
    python data_loader_ld2011.py

    if [ $? -eq 0 ]; then
        print_success "数据预处理完成"
    else
        print_error "数据预处理失败"
        exit 1
    fi
fi

################################################################################
# 步骤 2: 模型训练
################################################################################

print_header "步骤 2: 模型训练"

print_info "将在以下缺失率下训练模型: ${MISSING_RATIOS[@]}"
echo ""

# 询问是否开始训练
read -p "是否开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "跳过模型训练"
else
    for ratio in "${MISSING_RATIOS[@]}"; do
        ratio_percent=$(echo "$ratio * 100" | bc)
        ratio_percent=${ratio_percent%.*}

        print_header "训练模型 - 缺失率: ${ratio_percent}%"

        print_info "训练参数:"
        echo "  - 缺失率: ${ratio_percent}%"
        echo "  - 迭代次数: ${N_ITERS}"
        echo "  - GPU设备: ${GPU_DEVICE}"
        echo "  - 配置文件: ${CONFIG_FILE}"
        echo ""

        # 运行训练
        python train_ld2011.py \
            --config ${CONFIG_FILE} \
            --missing_ratio ${ratio} \
            --device ${GPU_DEVICE}

        if [ $? -eq 0 ]; then
            print_success "缺失率 ${ratio_percent}% 训练完成"
        else
            print_error "缺失率 ${ratio_percent}% 训练失败"
        fi

        echo ""
    done

    print_success "所有模型训练完成"
fi

################################################################################
# 步骤 3: 模型评估
################################################################################

print_header "步骤 3: 模型评估"

read -p "是否开始评估? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "跳过模型评估"
else
    print_info "开始评估所有缺失率的模型..."

    python evaluate_ld2011.py \
        --config ${CONFIG_FILE} \
        --eval_all \
        --device ${GPU_DEVICE}

    if [ $? -eq 0 ]; then
        print_success "模型评估完成"

        # 显示结果表格
        RESULTS_FILE="./results/ld2011/evaluation_results.csv"
        if [ -f "${RESULTS_FILE}" ]; then
            print_header "评估结果"
            cat ${RESULTS_FILE}
        fi
    else
        print_error "模型评估失败"
    fi
fi

################################################################################
# 完成
################################################################################

print_header "实验完成"

print_info "结果位置:"
echo "  - 模型检查点: ./results/ld2011/"
echo "  - 损失曲线: ./results/ld2011/*/training_loss_curve.png"
echo "  - 可视化结果: ./results/ld2011/*/visualizations/"
echo "  - 评估报告: ./results/ld2011/evaluation_results.csv"
echo ""

print_success "所有实验完成！"
