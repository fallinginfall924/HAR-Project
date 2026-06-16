import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ==========================================
# 1. 准备数据（直接复用你刚才采集的6组数据）
# ==========================================
raw_data = np.array([
    [16444, 368, 1680],
    [-16272, 336, -504],
    [-16292, -284, -280],
    [-16360, 892, 368],
    [516, -540, 16660],
    [1276, -136, -16252]
], dtype=float)

# 你的校准参数
bias = np.array([42.00, 176.00, 204.00])
scale = np.array([1.0011, 1.0000, 1.0044])

# 计算校准后的数据 (Raw - Bias) * Scale
calibrated_data = (raw_data - bias) * scale

# 计算合加速度模长 (Norm)
raw_norm = np.linalg.norm(raw_data, axis=1)
cal_norm = np.linalg.norm(calibrated_data, axis=1)

# ==========================================
# 2. 开始画图
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 图1：加速度计模长分布对比图 (核心拿分点) ---
axes[0].scatter(range(6), raw_norm, color='red', label='标定前 (Raw)', s=80, zorder=3)
axes[0].scatter(range(6), cal_norm, color='green', label='标定后 (Calibrated)', s=80, zorder=3)
axes[0].axhline(y=16384, color='blue', linestyle='--', label='理论标准值 (1g = 16384 LSB)')

axes[0].set_title('加速度计合加速度模长分布对比', fontsize=14)
axes[0].set_xlabel('六个静止姿态')
axes[0].set_ylabel('模长 (LSB)')
axes[0].set_xticks(range(6))
axes[0].set_xticklabels(['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'])
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# --- 图2：三轴数据校准前后对比柱状图 ---
x = np.arange(6)  # X轴位置标签
width = 0.25      # 柱子宽度

# 绘制标定前和标定后的 X/Y/Z 轴数据
bars1 = axes[1].bar(x - width, raw_data[:, 0], width, label='标定前 X', color='#ffcccc')
bars2 = axes[1].bar(x,         raw_data[:, 1], width, label='标定前 Y', color='#ffffcc')
bars3 = axes[1].bar(x + width, raw_data[:, 2], width, label='标定前 Z', color='#ccffcc')

# 为了图表清晰，这里只画标定后的主轴数据（即每个姿态下接近 ±1g 的那个轴）
# 你也可以根据需要展开画，但通常看模长图就足够了
axes[1].set_title('六位置原始数据分布概览', fontsize=14)
axes[1].set_xlabel('六个静止姿态')
axes[1].set_ylabel('传感器读数 (LSB)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'])
axes[1].legend(loc='upper right')
axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()

# ==========================================
# 3. 自动保存图片到 docs 文件夹
# ==========================================
save_path = '../docs/calibration_comparison.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=150)
print(f"🎉 图片已成功保存至: {save_path}")

plt.show()