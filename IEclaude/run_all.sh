#!/bin/bash
# ============================================================================
# IEclaude 完整训练和评估脚本
#
# 功能：
#   1. 生成所有配置文件
#   2. 依次训练所有缺失率的模型
#   3. 评估所有模型
#   4. 生成汇总报告
#
# 使用方法：
#   chmod +x run_all.sh
#   ./run_all.sh --gpu 0
# ============================================================================

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 解析命令行参数
GPU=0
if [ "$1" == "--gpu" ]; then
    GPU=$2
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}IEclaude 完整训练和评估流程${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "使用GPU: ${GPU}"
echo ""

# 设置GPU环境变量
export CUDA_VISIBLE_DEVICES=${GPU}

# 缺失率列表
MISSING_RATES=(20 30 40 50 60 70 80)

# ============================================================================
# 步骤1: 生成配置文件
# ============================================================================
echo -e "${YELLOW}步骤1: 生成配置文件${NC}"
python generate_configs.py
echo ""

# ============================================================================
# 步骤2: 训练所有模型
# ============================================================================
echo -e "${YELLOW}步骤2: 训练所有模型${NC}"

for RATE in "${MISSING_RATES[@]}"; do
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}训练缺失率 ${RATE}% 的模型${NC}"
    echo -e "${GREEN}========================================${NC}"

    python train.py \
        --config configs/config_${RATE}.json \
        --gpu ${GPU}

    if [ $? -ne 0 ]; then
        echo -e "${RED}训练失败: 缺失率 ${RATE}%${NC}"
        exit 1
    fi

    echo ""
done

# ============================================================================
# 步骤3: 评估所有模型
# ============================================================================
echo -e "${YELLOW}步骤3: 评估所有模型${NC}"

# 创建汇总结果文件
SUMMARY_FILE="results/summary_results.txt"
mkdir -p results
echo "Missing_Rate,MAE,RMSE" > ${SUMMARY_FILE}

for RATE in "${MISSING_RATES[@]}"; do
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}评估缺失率 ${RATE}% 的模型${NC}"
    echo -e "${GREEN}========================================${NC}"

    python evaluate.py \
        --config configs/config_${RATE}.json \
        --checkpoint results/traffic_${RATE}/best_model.pt \
        --gpu ${GPU} \
        --num_samples 10

    if [ $? -ne 0 ]; then
        echo -e "${RED}评估失败: 缺失率 ${RATE}%${NC}"
        exit 1
    fi

    # 提取指标并添加到汇总文件
    METRICS_FILE="results/traffic_${RATE}/evaluation_metrics.txt"
    if [ -f ${METRICS_FILE} ]; then
        MAE=$(grep "MAE:" ${METRICS_FILE} | awk '{print $2}')
        RMSE=$(grep "RMSE:" ${METRICS_FILE} | awk '{print $2}')
        echo "${RATE},${MAE},${RMSE}" >> ${SUMMARY_FILE}
    fi

    echo ""
done

# ============================================================================
# 步骤4: 生成汇总报告
# ============================================================================
echo -e "${YELLOW}步骤4: 生成汇总报告${NC}"

# 创建汇总报告
REPORT_FILE="results/final_report.txt"

cat > ${REPORT_FILE} << EOF
========================================
IEclaude 实验结果汇总
========================================

实验配置:
- 数据集: LD2011_2014.txt
- 序列长度: 168 (1周)
- 模型: 隐式显式扩散模型
  * 隐式模块: TCN (扩张率 [1,2,4,8])
  * 显式模块: S4 (状态维度 64, 4层)
- 扩散步数: 200
- 训练epochs: 100

实验结果:
========================================
Missing Rate | MAE    | RMSE
========================================
EOF

# 读取汇总文件并格式化输出
tail -n +2 ${SUMMARY_FILE} | while IFS=',' read -r RATE MAE RMSE; do
    printf "%-12s | %-6s | %-6s\n" "${RATE}%" "${MAE}" "${RMSE}" >> ${REPORT_FILE}
done

echo "========================================" >> ${REPORT_FILE}
echo "" >> ${REPORT_FILE}
echo "目标指标 (参考):" >> ${REPORT_FILE}
echo "========================================" >> ${REPORT_FILE}
echo "Missing Rate | MAE    | RMSE" >> ${REPORT_FILE}
echo "========================================" >> ${REPORT_FILE}
echo "20%          | 0.272  | 0.389" >> ${REPORT_FILE}
echo "30%          | 0.297  | 0.424" >> ${REPORT_FILE}
echo "40%          | 0.334  | 0.477" >> ${REPORT_FILE}
echo "50%          | 0.378  | 0.540" >> ${REPORT_FILE}
echo "60%          | 0.450  | 0.655" >> ${REPORT_FILE}
echo "70%          | 0.541  | 0.776" >> ${REPORT_FILE}
echo "80%          | 0.732  | 1.049" >> ${REPORT_FILE}
echo "========================================" >> ${REPORT_FILE}

# 显示汇总报告
cat ${REPORT_FILE}

# 保存汇总报告
echo -e "\n${GREEN}汇总报告已保存: ${REPORT_FILE}${NC}"

# ============================================================================
# 完成
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "结果位置: ./results/"
echo -e "汇总报告: ${REPORT_FILE}"
echo ""
