#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境验证脚本
用于检查所有依赖是否正确安装

运行此脚本确保环境配置正确
"""

import sys

print("=" * 80)
print("隐式-显式扩散模型 - 环境验证")
print("=" * 80)

# 检查Python版本
print(f"\n1. Python版本检查:")
print(f"   当前版本: {sys.version}")
if sys.version_info >= (3, 8):
    print("   ✅ Python版本符合要求 (>= 3.8)")
else:
    print("   ❌ Python版本过低，需要 >= 3.8")
    sys.exit(1)

# 检查核心依赖
print("\n2. 核心依赖检查:")

required_packages = {
    'torch': '深度学习框架',
    'numpy': '数值计算',
    'pandas': '数据处理',
    'matplotlib': '可视化',
    'sklearn': 'Scikit-learn机器学习',
    'scipy': '科学计算',
    'tqdm': '进度条',
    'einops': '张量操作',
    'opt_einsum': '优化的einsum'
}

missing_packages = []
installed_packages = {}

for package, description in required_packages.items():
    try:
        if package == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')

        print(f"   ✅ {package:15s} {version:10s} ({description})")
        installed_packages[package] = version
    except ImportError:
        print(f"   ❌ {package:15s} 未安装 ({description})")
        missing_packages.append(package)

# 检查CUDA
print("\n3. CUDA检查:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA可用")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            # 获取显存信息
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"          显存: {total_memory:.1f} GB")
    else:
        print("   ⚠️  CUDA不可用，将使用CPU训练（速度会很慢）")
except Exception as e:
    print(f"   ❌ CUDA检查失败: {e}")

# 检查S4模型依赖
print("\n4. S4模型依赖检查:")
try:
    from scipy import special as ss
    print("   ✅ scipy.special 可用")
except ImportError:
    print("   ❌ scipy.special 不可用")
    missing_packages.append('scipy.special')

# 总结
print("\n" + "=" * 80)
if len(missing_packages) == 0:
    print("✅ 所有依赖检查通过！环境配置正确。")
    print("\n下一步：")
    print("1. 确认数据文件路径")
    print("2. 运行数据预处理: python data_loader_ld2011.py")
    print("3. 开始训练模型")
else:
    print("❌ 缺少以下依赖包:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print("\n请运行以下命令安装缺少的包:")
    print(f"pip install {' '.join(missing_packages)}")

print("=" * 80)

# 简单的PyTorch测试
print("\n5. PyTorch功能测试:")
try:
    import torch

    # 创建张量
    x = torch.randn(2, 3)
    print(f"   ✅ 张量创建成功: shape {x.shape}")

    # 测试CUDA
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        print(f"   ✅ CUDA张量创建成功: device {x_cuda.device}")

    # 测试简单运算
    y = x * 2 + 1
    print(f"   ✅ 张量运算正常")

    print("\n✅ PyTorch环境测试通过！")

except Exception as e:
    print(f"\n❌ PyTorch测试失败: {e}")

print("\n" + "=" * 80)
