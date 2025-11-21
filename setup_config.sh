#!/bin/bash
# ============================================================================
# 交通数据插补隐式显式扩散模型配置脚本
# 用途：自动创建针对LD2011_2014.txt数据集的配置文件
# ============================================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}交通数据插补模型配置向导${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查是否在正确的目录
if [ ! -d "src" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi

# 创建配置目录（如果不存在）
CONFIG_DIR="src/config"
mkdir -p "$CONFIG_DIR"

# 询问用户GPU设置
echo -e "\n${YELLOW}请选择要使用的GPU:${NC}"
echo "1) cuda:0 (A40)"
echo "2) cuda:1 (A10)"
read -p "请输入选项 (1 或 2): " gpu_choice

case $gpu_choice in
    1)
        GPU_ID=0
        GPU_NAME="A40"
        ;;
    2)
        GPU_ID=1
        GPU_NAME="A10"
        ;;
    *)
        echo -e "${YELLOW}无效选择，默认使用 cuda:0 (A40)${NC}"
        GPU_ID=0
        GPU_NAME="A40"
        ;;
esac

echo -e "${GREEN}✓ 已选择 GPU: cuda:${GPU_ID} (${GPU_NAME})${NC}"

# 询问缺失率
echo -e "\n${YELLOW}请选择要训练的缺失率:${NC}"
echo "1) 20%"
echo "2) 30%"
echo "3) 40%"
echo "4) 50%"
echo "5) 60%"
echo "6) 70%"
echo "7) 80%"
echo "8) 全部 (依次训练所有缺失率)"
read -p "请输入选项 (1-8): " missing_choice

case $missing_choice in
    1) MISSING_RATES=(20) ;;
    2) MISSING_RATES=(30) ;;
    3) MISSING_RATES=(40) ;;
    4) MISSING_RATES=(50) ;;
    5) MISSING_RATES=(60) ;;
    6) MISSING_RATES=(70) ;;
    7) MISSING_RATES=(80) ;;
    8) MISSING_RATES=(20 30 40 50 60 70 80) ;;
    *)
        echo -e "${YELLOW}无效选择，默认使用 20%${NC}"
        MISSING_RATES=(20)
        ;;
esac

echo -e "${GREEN}✓ 已选择缺失率: ${MISSING_RATES[@]}%${NC}"

# 数据集路径
DATA_PATH="/home/zhu/sssdtcn/LD2011_2014.txt"

# 检查数据集是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}警告: 数据集文件不存在: $DATA_PATH${NC}"
    echo -e "${YELLOW}请确保数据集已正确放置${NC}"
fi

# 为每个缺失率创建配置文件
for MISSING_RATE in "${MISSING_RATES[@]}"; do
    echo -e "\n${YELLOW}正在创建缺失率 ${MISSING_RATE}% 的配置文件...${NC}"

    CONFIG_FILE="${CONFIG_DIR}/config_traffic_${MISSING_RATE}.json"

    cat > "$CONFIG_FILE" << EOF
{
    "diffusion_config": {
        "T": 200,
        "beta_0": 0.0001,
        "beta_T": 0.02
    },
    "wavenet_config": {
        "in_channels": 370,
        "out_channels": 370,
        "num_res_layers": 36,
        "res_channels": 256,
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 168,
        "s4_d_state": 64,
        "s4_dropout": 0.0,
        "s4_bidirectional": 1,
        "s4_layernorm": 1,
        "tcn_channels": [256, 256, 256],
        "tcn_kernel_size": 3,
        "tcn_dilation_rates": [1, 2, 4, 8]
    },
    "train_config": {
        "output_directory": "./results/traffic/${MISSING_RATE}",
        "ckpt_iter": "max",
        "iters_per_ckpt": 1000,
        "iters_per_logging": 100,
        "n_iters": 50000,
        "learning_rate": 2e-4,
        "batch_size": 8,
        "only_generate_missing": 1,
        "use_model": 2,
        "masking": "rm",
        "missing_k": ${MISSING_RATE},
        "gpu_id": ${GPU_ID}
    },
    "trainset_config": {
        "train_data_path": "${DATA_PATH}",
        "test_data_path": "${DATA_PATH}",
        "segment_length": 168,
        "sampling_rate": 168,
        "train_split": 0.8
    },
    "gen_config": {
        "output_directory": "./results/traffic/${MISSING_RATE}",
        "ckpt_path": "./results/traffic/${MISSING_RATE}/"
    }
}
EOF

    echo -e "${GREEN}✓ 已创建配置文件: $CONFIG_FILE${NC}"
done

# 创建环境变量设置脚本
ENV_FILE="set_gpu_env.sh"
cat > "$ENV_FILE" << EOF
#!/bin/bash
# GPU环境变量设置
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "已设置 CUDA_VISIBLE_DEVICES=${GPU_ID}"
EOF

chmod +x "$ENV_FILE"
echo -e "\n${GREEN}✓ 已创建GPU环境设置脚本: $ENV_FILE${NC}"

# 创建训练启动脚本
TRAIN_SCRIPT="run_training.sh"
cat > "$TRAIN_SCRIPT" << EOF
#!/bin/bash
# 交通数据插补模型训练脚本

# 设置GPU环境
source ./set_gpu_env.sh

# 切换到src目录
cd src

# 训练模型
for MISSING_RATE in ${MISSING_RATES[@]}; do
    echo "=========================================="
    echo "开始训练缺失率 \${MISSING_RATE}% 的模型"
    echo "=========================================="

    # 清理之前的结果
    rm -rf ./results/traffic/\${MISSING_RATE}/*

    # 开始训练
    python3 train.py -c config/config_traffic_\${MISSING_RATE}.json

    # 运行推理
    echo "开始推理和评估..."
    python3 inference.py -c config/config_traffic_\${MISSING_RATE}.json

    echo "缺失率 \${MISSING_RATE}% 训练完成！"
    echo ""
done

echo "所有训练任务完成！"
EOF

chmod +x "$TRAIN_SCRIPT"
echo -e "${GREEN}✓ 已创建训练启动脚本: $TRAIN_SCRIPT${NC}"

# 显示后续步骤
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}配置完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${YELLOW}后续步骤:${NC}"
echo -e "1. 安装依赖包:"
echo -e "   ${GREEN}cd src && pip install -r requirements.txt${NC}"
echo -e "\n2. 开始训练:"
echo -e "   ${GREEN}./run_training.sh${NC}"
echo -e "\n3. 查看结果:"
echo -e "   ${GREEN}ls -la results/traffic/${NC}"
echo -e "\n${YELLOW}提示:${NC}"
echo -e "- 配置文件位置: src/config/config_traffic_XX.json"
echo -e "- GPU设置: cuda:${GPU_ID} (${GPU_NAME})"
echo -e "- 数据集: ${DATA_PATH}"
echo ""
