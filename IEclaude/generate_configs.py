"""
生成所有缺失率的配置文件

使用方法：
    python generate_configs.py
"""

import json
import os

# 基础配置
base_config = {
    "seed": 42,
    "data": {
        "data_path": "/home/zhu/sssdtcn/LD2011_2014.txt",
        "seq_len": 168,
        "stride": 84,
        "train_ratio": 0.7,
        "normalize": "standard"
    },
    "model": {
        "res_channels": 256,
        "skip_channels": 256,
        "num_res_layers": 36,
        "dilation_cycle": 10,
        "tcn_channels": [256, 256, 256],
        "tcn_kernel_size": 3,
        "tcn_dilation_rates": [1, 2, 4, 8],
        "tcn_dropout": 0.0,
        "s4_d_state": 64,
        "s4_n_layers": 4,
        "s4_dropout": 0.0,
        "s4_bidirectional": True
    },
    "diffusion": {
        "T": 200,
        "beta_0": 0.0001,
        "beta_T": 0.02,
        "schedule": "linear",
        "embed_dim_in": 128,
        "embed_dim_mid": 512,
        "embed_dim_out": 512
    },
    "train": {
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 0.0002,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "min_lr": 1e-6,
        "save_interval": 10,
        "num_workers": 0,
        "only_generate_missing": True,
        "clean_before_train": True,
        "missing_pattern": "random"
    }
}

# 创建configs目录
os.makedirs('configs', exist_ok=True)

# 生成不同缺失率的配置文件
missing_rates = [20, 30, 40, 50, 60, 70, 80]

for rate in missing_rates:
    config = base_config.copy()
    config = json.loads(json.dumps(config))  # 深拷贝

    # 设置缺失率
    config['train']['missing_rate'] = rate / 100.0
    config['train']['output_dir'] = f'./results/traffic_{rate}'

    # 添加注释
    config['comment'] = f'IEclaude配置文件 - {rate}%缺失率'

    # 保存配置文件
    config_path = f'configs/config_{rate}.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f'已生成配置文件: {config_path}')

print(f'\n共生成 {len(missing_rates)} 个配置文件')
