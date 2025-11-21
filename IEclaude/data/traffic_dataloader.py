"""
交通数据加载器 (Traffic Data Loader)

功能说明：
    加载和预处理LD2011_2014.txt交通数据集，用于时间序列插补任务

数据集说明：
    - LD2011_2014.txt: 电力负载数据集
    - 包含370个客户的电力消耗时间序列
    - 采样频率: 每15分钟一个数据点
    - 时间跨度: 2011-2014年

预处理流程：
    1. 读取原始数据
    2. 数据归一化（标准化或最小-最大缩放）
    3. 切分为固定长度的序列段
    4. 生成训练/测试集
    5. 创建缺失mask（根据指定缺失率）

超参数调节指南：
    1. 数据处理参数：
       - seq_len: 序列长度 [建议: 168 for 1周]
       - stride: 滑动窗口步长 [建议: seq_len//2]
       - train_ratio: 训练集比例 [建议: 0.7-0.8]
       - normalize_method: 归一化方法 ['standard', 'minmax']

    2. 缺失模式：
       - missing_rate: 缺失率 [0.2-0.8]
       - missing_pattern: 缺失模式 ['random', 'block', 'spatial']
         * random: 随机缺失
         * block: 连续块缺失
         * spatial: 空间缺失（某些传感器整体缺失）

    3. 数据增强：
       - add_noise: 是否添加噪声
       - noise_std: 噪声标准差
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


class TrafficDataset(Dataset):
    """
    交通数据集类

    功能：
        - 存储预处理后的数据
        - 生成缺失mask
        - 返回训练样本

    参数：
        data (np.array): 数据 [num_samples, num_channels, seq_len]
        missing_rate (float): 缺失率
        missing_pattern (str): 缺失模式
    """
    def __init__(self, data, missing_rate=0.3, missing_pattern='random'):
        super().__init__()

        self.data = torch.FloatTensor(data)  # [N, C, L]
        self.missing_rate = missing_rate
        self.missing_pattern = missing_pattern

        self.num_samples, self.num_channels, self.seq_len = self.data.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取一个样本

        返回：
            sample (dict): 包含以下键值
                - 'observed_data': 观测数据 [C, L]
                - 'mask': 缺失mask [C, L], 1=观测，0=缺失
                - 'ground_truth': 真实数据 [C, L]
                - 'index': 样本索引
        """
        # 获取原始数据
        ground_truth = self.data[idx]  # [C, L]

        # 生成缺失mask
        mask = self._generate_mask(ground_truth.shape)

        # 应用mask生成观测数据
        observed_data = ground_truth * mask

        sample = {
            'observed_data': observed_data,
            'mask': mask,
            'ground_truth': ground_truth,
            'index': idx
        }

        return sample

    def _generate_mask(self, shape):
        """
        生成缺失mask

        参数：
            shape (tuple): 数据形状 [C, L]

        返回：
            mask (torch.Tensor): 缺失mask [C, L]
        """
        C, L = shape

        if self.missing_pattern == 'random':
            # 随机缺失：每个位置独立地以missing_rate的概率缺失
            mask = torch.rand(C, L) > self.missing_rate

        elif self.missing_pattern == 'block':
            # 块缺失：连续的时间段缺失
            mask = torch.ones(C, L)
            # 随机选择几个连续块进行缺失
            num_blocks = int(self.missing_rate * L / 10)  # 每个块大约10个时间步
            for _ in range(num_blocks):
                for c in range(C):
                    block_start = np.random.randint(0, L - 10)
                    block_len = np.random.randint(5, 15)
                    block_end = min(block_start + block_len, L)
                    mask[c, block_start:block_end] = 0

        elif self.missing_pattern == 'spatial':
            # 空间缺失：某些通道完全缺失
            mask = torch.ones(C, L)
            num_missing_channels = int(self.missing_rate * C)
            missing_channels = np.random.choice(C, num_missing_channels, replace=False)
            mask[missing_channels, :] = 0

        else:
            raise ValueError(f"未知的缺失模式: {self.missing_pattern}")

        return mask.float()


def load_traffic_data(
    data_path,
    seq_len=168,
    stride=84,
    train_ratio=0.7,
    normalize_method='standard',
    verbose=True
):
    """
    加载交通数据

    参数：
        data_path (str): 数据文件路径
        seq_len (int): 序列长度（时间步数）
        stride (int): 滑动窗口步长
        train_ratio (float): 训练集比例
        normalize_method (str): 归一化方法 ['standard', 'minmax', 'none']
        verbose (bool): 是否打印信息

    返回：
        train_data (np.array): 训练数据 [N_train, C, L]
        test_data (np.array): 测试数据 [N_test, C, L]
        scaler: 归一化器（用于反归一化）
        data_info (dict): 数据信息
    """
    if verbose:
        print("=" * 80)
        print("加载交通数据集")
        print("=" * 80)

    # ========================================================================
    # 步骤1: 读取原始数据
    # ========================================================================
    if verbose:
        print(f"\n1. 读取数据文件: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 读取数据
    # LD2011_2014.txt是以分号分隔的CSV文件
    # 第一列是时间戳，其余370列是客户数据
    try:
        # 尝试读取带分号的CSV
        df = pd.read_csv(data_path, sep=';', decimal=',')
    except:
        # 如果失败，尝试普通CSV
        df = pd.read_csv(data_path)

    if verbose:
        print(f"   原始数据形状: {df.shape}")
        print(f"   列名: {df.columns.tolist()[:5]}...")  # 显示前5个列名

    # 移除第一列（时间戳或索引）
    # 假设第一列不是数据列
    if df.shape[1] == 371:  # 1个时间戳 + 370个客户
        data = df.iloc[:, 1:].values.astype(np.float32)
    else:
        data = df.values.astype(np.float32)

    if verbose:
        print(f"   数据数组形状: {data.shape}  [时间步, 通道数]")

    # ========================================================================
    # 步骤2: 数据归一化
    # ========================================================================
    if verbose:
        print(f"\n2. 数据归一化 (方法: {normalize_method})")

    if normalize_method == 'standard':
        # 标准化：减去均值，除以标准差
        # 对每个通道独立标准化
        scaler = StandardScaler()
        data = scaler.fit_transform(data)  # [T, C]

    elif normalize_method == 'minmax':
        # 最小-最大缩放：缩放到[0, 1]
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)  # [T, C]

    elif normalize_method == 'none':
        # 不进行归一化
        scaler = None

    else:
        raise ValueError(f"未知的归一化方法: {normalize_method}")

    if verbose:
        print(f"   归一化后统计:")
        print(f"     均值: {data.mean():.4f}")
        print(f"     标准差: {data.std():.4f}")
        print(f"     最小值: {data.min():.4f}")
        print(f"     最大值: {data.max():.4f}")

    # ========================================================================
    # 步骤3: 切分为固定长度的序列段
    # ========================================================================
    if verbose:
        print(f"\n3. 切分序列 (seq_len={seq_len}, stride={stride})")

    T, C = data.shape  # T: 时间步数, C: 通道数（传感器数量）

    # 使用滑动窗口切分
    sequences = []
    for start in range(0, T - seq_len + 1, stride):
        end = start + seq_len
        seq = data[start:end, :]  # [seq_len, C]
        # 转置为 [C, seq_len] 格式（PyTorch卷积的标准格式）
        seq = seq.T  # [C, seq_len]
        sequences.append(seq)

    sequences = np.array(sequences)  # [N, C, seq_len]

    if verbose:
        print(f"   序列数量: {len(sequences)}")
        print(f"   序列形状: {sequences.shape}  [样本数, 通道数, 序列长度]")

    # ========================================================================
    # 步骤4: 切分训练集和测试集
    # ========================================================================
    if verbose:
        print(f"\n4. 切分训练/测试集 (训练比例: {train_ratio})")

    num_samples = len(sequences)
    num_train = int(num_samples * train_ratio)

    # 按时间顺序划分：前面的用于训练，后面的用于测试
    train_data = sequences[:num_train]
    test_data = sequences[num_train:]

    if verbose:
        print(f"   训练集: {train_data.shape}")
        print(f"   测试集: {test_data.shape}")

    # ========================================================================
    # 步骤5: 数据信息
    # ========================================================================
    data_info = {
        'num_channels': C,
        'seq_len': seq_len,
        'num_train': len(train_data),
        'num_test': len(test_data),
        'normalize_method': normalize_method,
        'original_shape': df.shape,
        'train_ratio': train_ratio,
        'stride': stride
    }

    if verbose:
        print("\n" + "=" * 80)
        print("数据加载完成！")
        print("=" * 80)

    return train_data, test_data, scaler, data_info


def create_dataloader(
    data,
    missing_rate=0.3,
    missing_pattern='random',
    batch_size=8,
    shuffle=True,
    num_workers=0
):
    """
    创建数据加载器

    参数：
        data (np.array): 数据 [N, C, L]
        missing_rate (float): 缺失率
        missing_pattern (str): 缺失模式
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载线程数

    返回：
        dataloader (DataLoader): PyTorch数据加载器
    """
    dataset = TrafficDataset(
        data=data,
        missing_rate=missing_rate,
        missing_pattern=missing_pattern
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )

    return dataloader


# ============================================================================
# 使用示例和调试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试数据加载器
    """
    print("=" * 80)
    print("交通数据加载器测试")
    print("=" * 80)

    # 数据文件路径（实际使用时需要修改为真实路径）
    data_path = "/home/zhu/sssdtcn/LD2011_2014.txt"

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"\n警告: 数据文件不存在: {data_path}")
        print("测试将使用模拟数据")

        # 创建模拟数据用于测试
        print("\n创建模拟数据...")
        num_timesteps = 10000
        num_channels = 370
        mock_data = np.random.randn(num_timesteps, num_channels).astype(np.float32)

        # 保存为临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            data_path = f.name
            # 写入CSV格式
            pd.DataFrame(mock_data).to_csv(f, index=False)

        print(f"模拟数据已保存到: {data_path}")

    # 加载数据
    train_data, test_data, scaler, data_info = load_traffic_data(
        data_path=data_path,
        seq_len=168,
        stride=84,
        train_ratio=0.7,
        normalize_method='standard',
        verbose=True
    )

    print(f"\n数据信息:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")

    # 创建数据加载器
    print(f"\n创建数据加载器...")
    train_loader = create_dataloader(
        data=train_data,
        missing_rate=0.3,
        missing_pattern='random',
        batch_size=8,
        shuffle=True
    )

    print(f"  训练集batch数量: {len(train_loader)}")

    # 测试获取一个batch
    print(f"\n获取一个batch测试:")
    for batch in train_loader:
        print(f"  observed_data: {batch['observed_data'].shape}")
        print(f"  mask: {batch['mask'].shape}")
        print(f"  ground_truth: {batch['ground_truth'].shape}")
        print(f"  实际缺失率: {1 - batch['mask'].mean():.2%}")
        break

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

    # 清理临时文件
    if 'mock_data' in locals():
        os.unlink(data_path)
        print(f"\n已清理临时文件: {data_path}")
