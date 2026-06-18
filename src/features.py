import os
import numpy as np
import pandas as pd
from scipy.signal import welch

# ================= 1. 特征提取工具函数 =================

def extract_features(window_data):
    """
    从单个窗口数据中提取时域和频域特征
    :param window_data: shape (N, 3) -> [AccX, AccY, AccZ]
    :return: dict of features
    """
    # 计算合加速度 (Magnitude)，消除设备佩戴方向的影响
    magnitude = np.sqrt(np.sum(window_data ** 2, axis=1))

    features = {}

    # --- A. 时域特征 (Time Domain) ---
    # 1. 均值 (Mean) - 反映直流分量/重力影响
    features['mean_x'] = np.mean(window_data[:, 0])
    features['mean_y'] = np.mean(window_data[:, 1])
    features['mean_z'] = np.mean(window_data[:, 2])

    # 2. 标准差 (Std) - 反映运动剧烈程度
    features['std_mag'] = np.std(magnitude)

    # 3. 均方根 (RMS) - 反映信号能量
    features['rms_mag'] = np.sqrt(np.mean(magnitude ** 2))

    # 4. 峰峰值 (Peak-to-Peak) - 反映振幅范围
    features['ptp_mag'] = np.ptp(magnitude)

    # 5. 偏度 (Skewness) - 区分上楼/下楼的关键指标
    # 手动计算偏度公式: E[(x-mu)^3] / sigma^3
    skew_val = np.mean(((magnitude - np.mean(magnitude)) / np.std(magnitude)) ** 3)
    features['skewness_mag'] = skew_val

    # 6. 峰度 (Kurtosis) - 反映冲击感 (跑步 vs 走路)
    kurt_val = np.mean(((magnitude - np.mean(magnitude)) / np.std(magnitude)) ** 4) - 3
    features['kurtosis_mag'] = kurt_val

    # 7. 过零率 (Zero Crossing Rate) - 粗略估计频率
    zero_crossings = np.sum(np.diff(np.sign(magnitude - np.mean(magnitude))) != 0)
    features['zcr_mag'] = zero_crossings / len(magnitude)

    # 8. 信号幅值面积 (SMA) - 总运动量
    features['sma'] = np.sum(np.abs(window_data), axis=0).sum() / len(window_data)

    # --- B. 频域特征 (Frequency Domain) ---
    # 假设采样率为 50Hz (根据你的数据实际情况调整，这里默认50Hz)
    fs = 50.0

    # 使用 Welch 方法计算功率谱密度 (PSD)
    f, Pxx_den = welch(magnitude, fs=fs, nperseg=min(256, len(magnitude)))

    # 9. 主频 (Dominant Frequency) - 对应步频
    max_idx = np.argmax(Pxx_den[1:]) + 1  # 忽略 0Hz 的直流分量
    features['dom_freq'] = f[max_idx]

    # 10. 频谱质心 (Spectral Centroid) - 频率分布重心
    if np.sum(Pxx_den) > 0:
        features['spectral_centroid'] = np.sum(f * Pxx_den) / np.sum(Pxx_den)
    else:
        features['spectral_centroid'] = 0

    # 11. 谱熵 (Spectral Entropy) - 规律性 (越规律熵越低)
    Pxx_norm = Pxx_den / np.sum(Pxx_den)
    Pxx_norm = Pxx_norm[Pxx_norm > 0]  # 避免 log(0)
    features['spectral_entropy'] = -np.sum(Pxx_norm * np.log2(Pxx_norm))

    # 12. 频带能量比 (Band Energy Ratio) - 低频能量占比 (0-3Hz)
    low_freq_mask = f <= 3.0
    total_energy = np.sum(Pxx_den)
    if total_energy > 0:
        features['low_freq_ratio'] = np.sum(Pxx_den[low_freq_mask]) / total_energy
    else:
        features['low_freq_ratio'] = 0

    return features


# ================= 2. 主执行流程 =================

if __name__ == "__main__":
    print("🔧 D4 特征工程启动...")

    # 路径配置
    input_path = "../data/windowed_dataset.csv"
    output_path = "../data/feature_matrix.csv"

    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到输入文件 {input_path}，请先运行 preprocess.py")
    else:
        # 1. 加载数据
        print(f"📂 正在加载数据: {input_path}")
        df = pd.read_csv(input_path)

        # 2. 遍历每个样本进行特征提取
        feature_list = []
        labels = []
        sample_ids = []

        # 假设 CSV 格式为: ID, Label, AccX_0, AccX_1, ... AccZ_N
        # 获取所有加速度列名 (排除 ID 和 Label)
        acc_cols = [c for c in df.columns if c.startswith('acc')]
        num_points = len(acc_cols) // 3  # 每个轴的点数

        print(f"⏳ 正在处理 {len(df)} 个样本...")

        for idx, row in df.iterrows():
            # 重构窗口数据 (N, 3)
            x = row[[c for c in acc_cols if 'x' in c]].values.astype(float)
            y = row[[c for c in acc_cols if 'y' in c]].values.astype(float)
            z = row[[c for c in acc_cols if 'z' in c]].values.astype(float)

            window_data = np.column_stack([x, y, z])

            # 提取特征
            feats = extract_features(window_data)
            feature_list.append(feats)
            labels.append(row['label'])
            sample_ids.append(row.name)

            if (idx + 1) % 50 == 0:
                print(f"   进度: {idx + 1}/{len(df)}")

        # 3. 构建特征矩阵 DataFrame
        df_features = pd.DataFrame(feature_list)
        df_features.insert(0, 'id', sample_ids)
        df_features.insert(1, 'label', labels)

        # 4. 保存结果
        df_features.to_csv(output_path, index=False)
        print("\n✅ 特征提取完成！")
        print(f"📊 输出文件: {output_path}")
        print(f"📐 矩阵维度: {df_features.shape} (样本数: {df_features.shape[0]}, 特征数: {df_features.shape[1]-2})")
        print("\n👇 前 5 行数据预览:")
        print(df_features.head())