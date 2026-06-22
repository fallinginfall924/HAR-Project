import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# 设置中文字体，防止画图时中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 核心工具函数 =================

def separate_gravity(acc_data, fs, cutoff=0.3):
    """
    使用巴特沃斯低通滤波器分离重力分量和运动分量
    """
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    gravity = filtfilt(b, a, acc_data, axis=0)
    motion = acc_data - gravity
    return gravity, motion


def sliding_window(data, labels, window_size, overlap=0.5):
    """
    对数据进行滑动窗口分割
    """
    step = int(window_size * (1 - overlap))
    windows, window_labels = [], []

    if len(data) < window_size:
        return np.array([]), np.array([])

    for start in range(0, len(data) - window_size + 1, step):
        end = start + window_size
        window_data = data[start:end]
        window_label_segment = labels[start:end]

        dominant_label = np.bincount(window_label_segment.astype(int)).argmax()
        if np.mean(window_label_segment == dominant_label) >= 0.8:
            windows.append(window_data)
            window_labels.append(dominant_label)

    return np.array(windows), np.array(window_labels)


def plot_waveform_comparison(acc_data, motion_acc, fs, save_path):
    """
    绘制预处理前后的波形对比图
    """
    time = np.arange(len(acc_data)) / fs
    plt.figure(figsize=(12, 8))

    # 原始加速度波形
    plt.subplot(2, 1, 1)
    plt.plot(time, acc_data)
    plt.title('预处理前：原始加速度波形 (含重力)', fontsize=12)
    plt.ylabel('加速度 (m/s^2)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 运动加速度波形
    plt.subplot(2, 1, 2)
    plt.plot(time, motion_acc, color='orange')
    plt.title('预处理后：纯运动加速度波形 (已分离重力)', fontsize=12)
    plt.xlabel('时间 (秒)')
    plt.ylabel('加速度 (m/s²)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   📊 波形对比图已保存至: {save_path}")


# ================= 主处理流程 (Pipeline) =================

def main():
    # --- 1. 参数配置 ---
    DATA_DIR = '../data/calibrated/'
    OUTPUT_FILE = '../data/windowed_dataset.csv'
    WAVEFORM_PLOT_PATH = '../data/preprocess_waveform_comparison.png'
    FS = 100  # 采样率 (Hz)

    # 标签映射字典
    label_mapping = {
        'RUNNING': 0, 'WALKING': 1, 'SITTING': 2,
        'STANDING': 3, 'STAIRS_UP': 4, 'STAIRS_DOWN': 5
    }

    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误：找不到数据文件夹 {DATA_DIR}，请检查路径！")
        return

    # --- 2. 【任务 S】生成预处理前后波形对比图 ---
    # 我们拿 RUNNING 文件来做波形对比演示
    sample_file = os.path.join(DATA_DIR, 'CALIB_S01_RUNNING_MOCK_01.csv')
    if os.path.exists(sample_file):
        print("🎨 正在生成预处理前后波形对比图...")
        df_sample = pd.read_csv(sample_file)
        acc_raw = df_sample[['acc_x', 'acc_y', 'acc_z']].values
        _, acc_motion = separate_gravity(acc_raw, FS)
        plot_waveform_comparison(acc_raw, acc_motion, FS, WAVEFORM_PLOT_PATH)
    else:
        print("⚠️ 未找到示例文件，跳过波形图生成。")

    # --- 3. 【任务 A】不同窗口长/重叠率的消融实验对比 ---
    print("\n🔬 开始消融实验：对比不同窗口参数对样本数量的影响...")
    experiments = [
        {"name": "基准配置", "window_sec": 2.56, "overlap": 0.5},
        {"name": "短窗口+低重叠", "window_sec": 1.28, "overlap": 0.25},
        {"name": "长窗口+高重叠", "window_sec": 5.12, "overlap": 0.75},
        {"name": "基准+高重叠", "window_sec": 2.56, "overlap": 0.75},
    ]

    exp_results = []
    for exp in experiments:
        total_samples = 0
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(DATA_DIR, filename))
                acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
                act_labels_str = df['activity_label'].values

                try:
                    act_labels_num = np.array([label_mapping[l] for l in act_labels_str])
                except KeyError:
                    continue

                _, motion_acc = separate_gravity(acc_data, FS)
                win_size = int(exp["window_sec"] * FS)
                windows, _ = sliding_window(motion_acc, act_labels_num, win_size, exp["overlap"])
                total_samples += len(windows)

        exp_results.append((exp["name"], exp["window_sec"], exp["overlap"], total_samples))
        print(f"   📌 {exp['name']: <10} | 窗口: {exp['window_sec']}s, 重叠: {exp['overlap']} | 样本数: {total_samples}")

    # 打印对比总结表格
    print("\n" + "=" * 65)
    print(f"{'实验配置': <12} | {'窗口长度(s)': <12} | {'重叠率': <8} | {'生成样本数'}")
    print("-" * 65)
    for res in exp_results:
        print(f"{res[0]: <12} | {res[1]: <12} | {res[2]: <8} | {res[3]}")
    print("=" * 65)

    # --- 4. 【任务 B】使用基准参数生成最终数据集 ---
    print("\n🏁 使用基准参数生成最终数据集...")
    WINDOW_SIZE_SEC = 2.56
    OVERLAP = 0.5

    all_windows, all_labels, all_subjects = [], [], []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_DIR, filename))
            acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
            act_labels_str = df['activity_label'].values
            subject_id = df['subject_id'].iloc[0]  # 提取被试者编号

            try:
                act_labels_num = np.array([label_mapping[l] for l in act_labels_str])
            except KeyError as e:
                print(f"   ❌ 错误：未知标签 '{e}'")
                continue

            _, motion_acc = separate_gravity(acc_data, FS)
            window_size = int(WINDOW_SIZE_SEC * FS)
            windows, win_labels = sliding_window(motion_acc, act_labels_num, window_size, OVERLAP)

            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(win_labels)
                all_subjects.extend([subject_id] * len(windows))  # 每个窗口对应一个subject_id

    if not all_windows:
        print("❌ 没有提取到任何有效窗口！")
        return

    final_windows = np.concatenate(all_windows, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    num_samples, win_size, num_channels = final_windows.shape
    flattened_windows = final_windows.reshape(num_samples, win_size * num_channels)

    column_names = [f'{ch}_{i}' for i in range(win_size) for ch in ['acc_x', 'acc_y', 'acc_z']]
    df_final = pd.DataFrame(flattened_windows, columns=column_names)
    df_final['label'] = final_labels
    df_final['subject_id'] = all_subjects  # 新增：保存被试者编号

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"\n🎉 最终数据集已保存至: {OUTPUT_FILE}")
    print(f"📊 总样本数: {len(df_final)}, 📐 特征维度: {len(column_names)}")
    print(f"👥 包含被试者: {df_final['subject_id'].unique()}")

if __name__ == '__main__':
    main()