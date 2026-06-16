import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import csv

# ==========================================
# 1. 从 CSV 文件读取数据
# ==========================================
mag_data = []
try:
    with open('mag_data.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3:
                mag_data.append((int(row[0]), int(row[1]), int(row[2])))
    print(f"✅ 成功从文件读取 {len(mag_data)} 组数据")
except FileNotFoundError:
    print("❌ 找不到 mag_data.csv 文件，请确认文件与脚本在同一目录下！")
    exit()

raw_data = np.array(mag_data, dtype=np.float64)


# ==========================================
# 2. 椭球拟合算法 (计算硬铁偏置和软铁矩阵)
# ==========================================
def ellipsoid_fit(data):
    """使用最小二乘法拟合椭球方程"""
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # 构建设计矩阵 D
    D = np.column_stack((x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z, np.ones_like(x)))

    # 求解最小二乘问题
    result = np.linalg.lstsq(D, np.ones_like(x, dtype=np.float64), rcond=None)[0]

    # 从结果中提取椭球参数并转换为硬铁偏置和软铁缩放因子
    # 这里采用简化的主轴对齐模型计算
    A = np.array([
        [result[0], result[5], result[4]],
        [result[5], result[1], result[3]],
        [result[4], result[3], result[2]]
    ])
    b = np.array([result[6], result[7], result[8]])

    # 计算中心点 (硬铁偏置)
    center = -np.linalg.solve(A, b)

    # 计算缩放因子 (软铁校正)
    # 为了简化显示，这里计算各轴半径的比例作为缩放矩阵的对角线
    radii = np.sqrt(np.abs(1.0 / np.diag(A)))
    scale_matrix = np.diag(radii / np.mean(radii))

    return center, scale_matrix


hard_iron_offset, soft_iron_scale = ellipsoid_fit(raw_data)

print("\n🎯 【校准参数生成完毕】 🎯")
print(
    f"硬铁偏置 (Hard Iron Offset): X={hard_iron_offset[0]:.2f}, Y={hard_iron_offset[1]:.2f}, Z={hard_iron_offset[2]:.2f}")
print(f"软铁缩放 (Soft Iron Scale Matrix):\n{soft_iron_scale}")

# ==========================================
# 3. 应用校准参数
# ==========================================
calibrated_data = np.dot((raw_data - hard_iron_offset), soft_iron_scale.T)

# ==========================================
# 4. 绘制标定前后对比图
# ==========================================
fig = plt.figure(figsize=(14, 6))

# 绘制原始数据
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(raw_data[:, 0], raw_data[:, 1], raw_data[:, 2], s=2, c='red', alpha=0.6)
ax1.set_title('Raw Magnetometer Data', fontsize=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 绘制校准后数据
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(calibrated_data[:, 0], calibrated_data[:, 1], calibrated_data[:, 2], s=2, c='green', alpha=0.6)
ax2.set_title('Calibrated Magnetometer Data', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()