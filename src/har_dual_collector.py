import json
import os
import numpy as np
import pandas as pd
import glob

# ================= 【配置区】 =================
# ⚠️【必改】输入文件夹：存放所有原始数据CSV文件的文件夹路径
INPUT_FOLDER = 'D:/HAR-Project/data/raw'
# ⚠️【必改】输出文件夹：校准后数据保存的文件夹路径
OUTPUT_FOLDER = 'D:/HAR-Project/data/calibrated'
# 标定参数文件路径
CALIB_FILE = 'calib_params.json'


# ==============================================

def load_calibration():
    """加载标定参数"""
    if not os.path.exists(CALIB_FILE):
        print(f"❌ 错误：找不到标定参数文件 '{CALIB_FILE}'！")
        return None
    with open(CALIB_FILE, 'r', encoding='utf-8') as f:
        print(f"✅ 成功加载标定参数文件: {CALIB_FILE}")
        return json.load(f)


def apply_calibration(raw_data, calib_params):
    """
    将一行原始数据 (list) 转换为校准后的数据 (list)
    """
    if not calib_params:
        return raw_data

    acc_raw = np.array(raw_data[0:3])
    gyro_raw = np.array(raw_data[3:6])

    # --- 1. 加速度计校准 ---
    acc_bias = np.array(calib_params['accelerometer']['bias'])
    acc_scale = np.array(calib_params['accelerometer']['scale_factor'])
    sensitivity_acc = calib_params['accelerometer'].get('sensitivity_lsb_per_g', 16384.0)
    acc_calibrated = ((acc_raw - acc_bias) / acc_scale) / sensitivity_acc

    # --- 2. 陀螺仪校准 ---
    # ⚠️【关键修改】键名 'gyro' 与您提供的 JSON 文件保持一致
    gyro_bias = np.array(calib_params['gyro']['static_bias'])
    sensitivity_gyro = 131.0
    gyro_calibrated = (gyro_raw - gyro_bias) / sensitivity_gyro

    # --- 3. 磁力计 (暂不处理) ---
    mag_calibrated = raw_data[6:9]

    return list(acc_calibrated) + list(gyro_calibrated) + list(mag_calibrated)


def process_single_file(input_filepath, output_filepath, calib_params):
    """处理单个文件的函数"""
    try:
        df_raw = pd.read_csv(input_filepath)
    except Exception as e:
        print(f"   ❌ 读取文件失败: {e}")
        return

    # 准备输出目录
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    header = "timestamp,subject_id,activity_label,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z"

    with open(output_filepath, 'w') as f:
        f.write(header + '\n')

        count = 0
        for _, row in df_raw.iterrows():
            raw_sensor_data = row[
                ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']].tolist()
            calibrated_data = apply_calibration(raw_sensor_data, calib_params)
            base_info = f"{row['timestamp']},{row['subject_id']},{row['activity_label']}"
            f.write(f"{base_info}," + ",".join(map(str, calibrated_data)) + "\n")
            count += 1
    print(f"   ✅ 完成: {os.path.basename(input_filepath)} (处理了 {count} 条数据)")


if __name__ == "__main__":
    # 1. 加载标定参数
    calib_params = load_calibration()
    if calib_params is None:
        exit()

    # 2. 查找所有需要处理的文件
    # 使用 glob 查找输入文件夹下所有以 RAW_ 开头，以 .csv 结尾的文件
    search_pattern = os.path.join(INPUT_FOLDER, 'RAW_*.csv')
    file_list = glob.glob(search_pattern)

    if not file_list:
        print(f"⚠️ 在 '{INPUT_FOLDER}' 中未找到任何以 'RAW_*.csv' 命名的文件。")
        print(f"   请检查路径是否正确，或文件是否存在。")
        exit()

    print(f"🔍 在 '{INPUT_FOLDER}' 中找到 {len(file_list)} 个文件，开始批量处理...")

    # 3. 逐个处理文件
    for i, input_file in enumerate(file_list):
        # 构造输出文件路径
        # 例如：输入是 D:/.../raw_data/RAW_S01_...csv
        # 输出是 D:/.../calib_data/CALIB_S01_...csv
        filename = os.path.basename(input_file)
        output_filename = filename.replace('RAW_', 'CALIB_', 1)
        output_file = os.path.join(OUTPUT_FOLDER, output_filename)

        print(f"\n[{i + 1}/{len(file_list)}] 正在处理: {filename}")
        process_single_file(input_file, output_file, calib_params)

    print("\n🎉 所有文件处理完毕！")