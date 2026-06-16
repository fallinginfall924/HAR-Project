import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


def allan_variance(data, fs):
    """计算单轴数据的 Allan 偏差"""
    N = len(data)
    max_m = int(np.floor(N / 2))
    if max_m < 2: return np.array([]), np.array([])

    m = np.unique(np.logspace(0, np.log10(max_m), num=100).astype(int))
    taus = m / fs
    adevs = np.zeros_like(taus, dtype=float)

    for i, cluster_size in enumerate(m):
        n_bins = N // cluster_size
        if n_bins < 2: continue
        reshaped_data = data[:n_bins * cluster_size].reshape(n_bins, cluster_size)
        means = np.mean(reshaped_data, axis=1)
        diff = np.diff(means)
        adevs[i] = np.sqrt(np.mean(diff ** 2) / 2.0)

    return taus, adevs


def analyze_gyro_static(filepath, fs=100):
    print(f"📂 正在读取文件: {filepath}")
    try:
        # header=None 表示没有表头，直接读数据
        df = pd.read_csv(filepath, header=None)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 检查是否有数据
    if df.empty or df.shape[1] == 0:
        print("❌ 文件为空！")
        return

    # === 关键处理：针对单列 LSB 数据 ===
    # 假设第一列 (索引0) 是我们要分析的轴 (比如 X 轴)
    raw_col_0 = df.iloc[:, 0].values.astype(float)

    # MPU6050 默认 ±2g/±250dps 量程下的灵敏度
    # 如果是 ±250dps，灵敏度通常是 131.0 LSB/(deg/s)
    SENSITIVITY = 131.0

    # 将原始 LSB 转换为 deg/s
    data_deg_s = raw_col_0 / SENSITIVITY

    print(f"✅ 读取到 {len(data_deg_s)} 个数据点 (已转换为 deg/s)")

    # --- 1. 计算静态零偏 (Static Bias) ---
    # 这就是你要填进 JSON 的值！
    static_bias_val = np.mean(data_deg_s)
    print(f"\n🎯 [重要] 算出的静态零偏 (X轴): {static_bias_val:.6f} deg/s")
    print(f"   (对应原始 LSB 均值: {np.mean(raw_col_0):.2f})")

    # --- 2. 计算 Allan 方差 ---
    # 去趋势：减去均值即可，防止直流分量影响
    data_detrend = data_deg_s - static_bias_val

    print("⏳ 正在计算 Allan 方差...")
    taus, adevs = allan_variance(data_detrend, fs)

    # 过滤无效点
    valid_idx = ~np.isnan(adevs) & (adevs > 0)
    taus = taus[valid_idx]
    adevs = adevs[valid_idx]

    if len(taus) == 0:
        print("❌ 计算结果为空，可能是数据量太少或采样率设置错误。")
        return

    # 找 BI (最小值)
    min_idx = np.argmin(adevs)
    bias_instability = adevs[min_idx]
    tau_bi = taus[min_idx]

    # --- 3. 绘图与保存 ---
    plt.figure(figsize=(10, 6))
    plt.loglog(taus, adevs, 'b-', linewidth=2, label='Allan Deviation')
    plt.loglog(tau_bi, bias_instability, 'ro', markersize=8,
               label=f'BI: {bias_instability:.6f} deg/s')

    plt.title('Gyroscope Allan Deviation (Single Axis)', fontsize=14)
    plt.xlabel('Averaging Time τ (s)', fontsize=12)
    plt.ylabel('Allan Deviation σ(τ)', fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.legend()

    output_dir = r"D:\HAR-Project\data"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"gyro_allan_single_axis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📊 结果汇总:")
    print(f"   - 零偏不稳定性 (BI): {bias_instability:.6f} deg/s")
    print(f"   - 建议填入 JSON 的 static_bias (X轴): [{static_bias_val:.4f}, 0.0, 0.0]")
    print(f"   - 图片已保存至: {save_path}")


if __name__ == "__main__":
    gyro_file = r"D:\HAR-Project\data\gyro_static_test.csv"
    SAMPLE_RATE = 100  # 确保这里和你采集时的频率一致
    analyze_gyro_static(gyro_file, fs=SAMPLE_RATE)