"""
LD2011_2014数据集加载和预处理脚本

LD2011_2014是一个电力消耗数据集，包含多个客户的用电量时间序列
该脚本负责：
1. 读取和解析原始txt文件
2. 数据归一化处理
3. 创建训练/测试分割
4. 生成不同缺失率的掩码

作者: Claude AI
日期: 2025-11-19
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import os


class LD2011Dataset:
    """
    LD2011_2014数据集处理类

    功能：
    1. 加载原始数据
    2. 数据清洗和预处理
    3. 归一化
    4. 分割训练集和测试集
    5. 生成序列窗口

    参数:
        data_path: 数据文件路径
        window_size: 滑动窗口大小，每个样本的序列长度
        stride: 滑动窗口的步长
        train_ratio: 训练集比例（剩余部分作为测试集）
        normalize: 是否进行标准化
        selected_features: 选择的特征数量（列数），None表示使用所有特征
    """
    def __init__(self, data_path, window_size=100, stride=50,
                 train_ratio=0.8, normalize=True, selected_features=None):
        """
        初始化数据集

        参数说明：
            data_path: 原始数据文件路径 (LD2011_2014.txt)
            window_size: 时间窗口大小，推荐值: 50-200
                        - 更大的窗口可以捕获更长期的依赖
                        - 更小的窗口计算更快，适合短期预测
            stride: 滑动窗口步长，推荐值: window_size // 2
                   - 更小的步长会产生更多样本（数据增强）
                   - 更大的步长样本间重叠更少
            train_ratio: 训练集占比，推荐值: 0.7-0.8
            normalize: 是否标准化，建议设为True
            selected_features: 选择前N个特征，None表示全部使用
        """
        self.data_path = data_path
        self.window_size = window_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.selected_features = selected_features

        # 用于归一化的scaler
        self.scaler = StandardScaler()

        # 加载数据
        print(f"正在从 {data_path} 加载数据...")
        self.raw_data = self._load_data()
        print(f"原始数据形状: {self.raw_data.shape}")

        # 预处理
        self.processed_data = self._preprocess()
        print(f"预处理后数据形状: {self.processed_data.shape}")

        # 分割训练集和测试集
        self.train_data, self.test_data = self._split_train_test()
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.test_data.shape}")

        # 创建滑动窗口序列
        self.train_sequences = self._create_sequences(self.train_data)
        self.test_sequences = self._create_sequences(self.test_data)
        print(f"训练序列数: {len(self.train_sequences)}")
        print(f"测试序列数: {len(self.test_sequences)}")

    def _load_data(self):
        """
        加载原始数据文件

        LD2011_2014.txt格式通常为：
        - 以分号或制表符分隔
        - 第一行可能是列名
        - 每行是一个时间点的数据

        返回:
            numpy数组，形状 [时间步数, 特征数]
        """
        try:
            # 尝试不同的分隔符
            # 首先尝试分号
            try:
                df = pd.read_csv(self.data_path, sep=';', decimal=',')
            except:
                # 如果失败，尝试制表符
                try:
                    df = pd.read_csv(self.data_path, sep='\t')
                except:
                    # 最后尝试逗号
                    df = pd.read_csv(self.data_path, sep=',')

            # 移除可能的时间戳列（通常是第一列）
            # 查找数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]

            # 如果指定了特征数量，选择前N个特征
            if self.selected_features is not None:
                n_features = min(self.selected_features, len(df.columns))
                df = df.iloc[:, :n_features]

            # 转换为numpy数组
            data = df.values

            # 处理缺失值和异常值
            # 用列均值填充NaN
            col_mean = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_mean, inds[1])

            return data

        except Exception as e:
            print(f"加载数据时出错: {e}")
            print("尝试使用备用加载方法...")

            # 备用方法：直接读取为文本
            with open(self.data_path, 'r') as f:
                lines = f.readlines()

            # 跳过表头
            data_lines = [line.strip() for line in lines[1:] if line.strip()]

            # 解析数据
            data = []
            for line in data_lines:
                # 尝试不同的分隔符
                if ';' in line:
                    values = line.split(';')
                elif '\t' in line:
                    values = line.split('\t')
                else:
                    values = line.split(',')

                # 转换为浮点数，跳过非数值列
                numeric_values = []
                for val in values:
                    try:
                        # 替换逗号为点（欧洲数值格式）
                        val = val.replace(',', '.')
                        numeric_values.append(float(val))
                    except:
                        continue

                if len(numeric_values) > 0:
                    data.append(numeric_values)

            data = np.array(data)

            # 如果指定了特征数量
            if self.selected_features is not None:
                n_features = min(self.selected_features, data.shape[1])
                data = data[:, :n_features]

            return data

    def _preprocess(self):
        """
        数据预处理

        步骤:
        1. 处理异常值（过大或过小的值）
        2. 标准化（零均值，单位方差）

        返回:
            预处理后的数据
        """
        data = self.raw_data.copy()

        # 1. 异常值处理：使用3-sigma原则
        # 超过3个标准差的值视为异常值，用边界值替换
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        data = np.clip(data, lower_bound, upper_bound)

        # 2. 标准化
        if self.normalize:
            data = self.scaler.fit_transform(data)

        return data

    def _split_train_test(self):
        """
        分割训练集和测试集

        使用时间顺序分割：
        - 前train_ratio的数据作为训练集
        - 后1-train_ratio的数据作为测试集

        这种分割方式符合时序数据的特点

        返回:
            (train_data, test_data)
        """
        n_samples = len(self.processed_data)
        split_idx = int(n_samples * self.train_ratio)

        train_data = self.processed_data[:split_idx]
        test_data = self.processed_data[split_idx:]

        return train_data, test_data

    def _create_sequences(self, data):
        """
        使用滑动窗口创建序列样本

        参数:
            data: 输入数据，形状 [时间步数, 特征数]

        返回:
            序列列表，每个序列形状 [window_size, 特征数]
        """
        sequences = []
        n_samples = len(data)

        # 滑动窗口提取序列
        for i in range(0, n_samples - self.window_size + 1, self.stride):
            seq = data[i:i + self.window_size]
            sequences.append(seq)

        return np.array(sequences)

    def get_train_data(self):
        """
        获取训练数据

        返回:
            训练序列，形状 [样本数, window_size, 特征数]
        """
        return self.train_sequences

    def get_test_data(self):
        """
        获取测试数据

        返回:
            测试序列，形状 [样本数, window_size, 特征数]
        """
        return self.test_sequences

    def inverse_transform(self, data):
        """
        反归一化

        将标准化后的数据转换回原始尺度

        参数:
            data: 标准化后的数据

        返回:
            原始尺度的数据
        """
        if self.normalize:
            # 如果输入是3D张量 [batch, length, features]
            if len(data.shape) == 3:
                batch_size, length, features = data.shape
                data_2d = data.reshape(-1, features)
                inverse_data = self.scaler.inverse_transform(data_2d)
                return inverse_data.reshape(batch_size, length, features)
            else:
                return self.scaler.inverse_transform(data)
        return data


def create_missing_mask(data, missing_ratio):
    """
    创建随机缺失掩码

    参数:
        data: 输入数据，形状 [batch, length, features]
        missing_ratio: 缺失率，例如0.3表示30%的数据缺失

    返回:
        mask: 二值掩码，形状与data相同
             1表示观测值，0表示缺失值
    """
    # 创建全1掩码
    mask = np.ones_like(data)

    batch_size, length, features = data.shape

    # 对每个特征独立生成掩码
    for b in range(batch_size):
        for f in range(features):
            # 计算该特征需要缺失的点数
            n_missing = int(length * missing_ratio)

            # 随机选择缺失位置
            missing_indices = np.random.choice(length, size=n_missing, replace=False)

            # 设置掩码
            mask[b, missing_indices, f] = 0

    return mask


def prepare_data_for_training(data_path, output_dir, window_size=100,
                              stride=50, selected_features=14,
                              train_ratio=0.8):
    """
    准备训练数据

    这个函数完成以下操作：
    1. 加载和预处理LD2011数据集
    2. 创建训练/测试分割
    3. 保存为numpy数组格式

    参数:
        data_path: 原始数据文件路径
        output_dir: 输出目录
        window_size: 时间窗口大小
        stride: 滑动窗口步长
        selected_features: 选择的特征数量
        train_ratio: 训练集比例

    返回:
        dataset对象
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    dataset = LD2011Dataset(
        data_path=data_path,
        window_size=window_size,
        stride=stride,
        train_ratio=train_ratio,
        normalize=True,
        selected_features=selected_features
    )

    # 获取训练和测试数据
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    # 保存为npy格式
    train_save_path = os.path.join(output_dir, 'train_ld2011.npy')
    test_save_path = os.path.join(output_dir, 'test_ld2011.npy')

    np.save(train_save_path, train_data)
    np.save(test_save_path, test_data)

    print(f"\n数据预处理完成！")
    print(f"训练数据保存至: {train_save_path}")
    print(f"测试数据保存至: {test_save_path}")
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")

    return dataset


if __name__ == "__main__":
    """
    使用示例
    """
    # 设置数据路径
    data_path = "/home/zhu/sssdtcn/LD2011_2014.txt"
    output_dir = "/home/user/SSSD/datasets"

    # 准备数据
    dataset = prepare_data_for_training(
        data_path=data_path,
        output_dir=output_dir,
        window_size=100,      # 序列长度100
        stride=50,            # 步长50
        selected_features=14, # 使用14个特征
        train_ratio=0.8       # 80%训练，20%测试
    )

    print("\n数据统计信息：")
    train_data = dataset.get_train_data()
    print(f"训练数据均值: {np.mean(train_data):.4f}")
    print(f"训练数据标准差: {np.std(train_data):.4f}")
    print(f"训练数据最小值: {np.min(train_data):.4f}")
    print(f"训练数据最大值: {np.max(train_data):.4f}")
