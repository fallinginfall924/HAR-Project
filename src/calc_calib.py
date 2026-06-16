import numpy as np

# 你在 ESP32 上采集到的 6 组完美数据
data = [
    [16444, 368, 1680],
    [-16272, 336, -504],
    [-16292, -284, -280],
    [-16360, 892, 368],
    [516, -540, 16660],
    [1276, -136, -16252]
]

# 将数据转为 numpy 数组
raw_data = np.array(data, dtype=float)

# 计算 X, Y, Z 轴的零偏 (Bias) = (Max + Min) / 2
bias_x = (max(raw_data[:, 0]) + min(raw_data[:, 0])) / 2.0
bias_y = (max(raw_data[:, 1]) + min(raw_data[:, 1])) / 2.0
bias_z = (max(raw_data[:, 2]) + min(raw_data[:, 2])) / 2.0

# 计算 X, Y, Z 轴的比例因子 (Scale Factor) = (Max - Min) / 2 / 1g(16384)
scale_x = (max(raw_data[:, 0]) - min(raw_data[:, 0])) / 2.0 / 16384.0
scale_y = (max(raw_data[:, 1]) - min(raw_data[:, 1])) / 2.0 / 16384.0
scale_z = (max(raw_data[:, 2]) - min(raw_data[:, 2])) / 2.0 / 16384.0

print("========================================")
print("🎉 MPU6050 加速度计校准参数计算完成！")
print("========================================")
print(f"X轴 - Bias: {bias_x:.2f}, Scale Factor: {scale_x:.4f}")
print(f"Y轴 - Bias: {bias_y:.2f}, Scale Factor: {scale_y:.4f}")
print(f"Z轴 - Bias: {bias_z:.2f}, Scale Factor: {scale_z:.4f}")
print("========================================")
print("\n💡 提示：在以后的代码中读取到原始数据 raw_ax 后，")
print("只需执行：calibrated_ax = (raw_ax - bias_x) * scale_x")
print("即可得到完美的校准数据！")