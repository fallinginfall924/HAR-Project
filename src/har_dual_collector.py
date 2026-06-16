import serial
import json
import time
import os
import numpy as np

# ================= 【配置区】 =================
COM_PORT = 'COM3'  # ⚠️【必改】你的 ESP32 串口号
BAUD_RATE = 115200  # ⚠️【必改】波特率
CALIB_FILE = 'calib_params.json'


# ==============================================

def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print(f"❌ 找不到 {CALIB_FILE}！将只保存原始数据。")
        return None
    with open(CALIB_FILE, 'r') as f:
        print("✅ 成功加载标定参数文件")
        return json.load(f)


def apply_calibration(raw_data, calib_params):
    """将原始 LSB 转换为物理单位"""
    if not calib_params: return raw_data

    acc_raw = np.array(raw_data[0:3])
    gyro_raw = np.array(raw_data[3:6])

    # 加速度计转 g
    acc_bias = np.array(calib_params['accelerometer']['bias'])
    acc_scale = np.array(calib_params['accelerometer']['scale_factor'])
    sensitivity_acc = calib_params['accelerometer'].get('sensitivity_lsb_per_g', 16384.0)
    acc_calibrated = ((acc_raw - acc_bias) / acc_scale) / sensitivity_acc

    # 陀螺仪转 deg/s
    gyro_bias = np.array(calib_params['gyroscope']['static_bias'])
    sensitivity_gyro = 131.0
    gyro_calibrated = (gyro_raw - gyro_bias) / sensitivity_gyro

    # 磁力计(暂不处理)
    mag_calibrated = raw_data[6:9]

    return list(acc_calibrated) + list(gyro_calibrated) + list(mag_calibrated)


if __name__ == "__main__":
    calib_params = load_calibration()
    os.makedirs('har_dataset', exist_ok=True)

    # --- 1. 交互式输入信息 ---
    subject_id = input("👤 请输入被试编号 (如 S01): ").strip()
    activity_label = input("🏷️ 请输入当前活动标签 (如 WALKING): ").strip()

    timestamp_str = str(int(time.time()))

    # 定义两个文件名
    raw_filename = f"har_dataset/RAW_{subject_id}_{activity_label}_{timestamp_str}.csv"
    calib_filename = f"har_dataset/CALIB_{subject_id}_{activity_label}_{timestamp_str}.csv"

    print(f"\n💾 即将生成两份数据:")
    print(f"   [Raw]   -> {raw_filename}")
    print(f"   [Calib] -> {calib_filename}")
    print("\n🎬 准备就绪！按回车键 [Enter] 开始采集，再按一次 [Ctrl+C] 停止...")
    input()

    # --- 2. 打开串口与两个 CSV 文件 ---
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    raw_file = open(raw_filename, 'w')
    calib_file = open(calib_filename, 'w')

    # 写入表头
    header = "timestamp,subject_id,activity_label,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z\n"
    raw_file.write(header)
    calib_file.write(header)

    count = 0
    print("▶️ 正在双轨采集...")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue

            parts = line.split(',')
            if len(parts) >= 9:
                try:
                    raw_data = [float(x) for x in parts[:9]]
                    ts = time.time()
                    base_info = f"{ts},{subject_id},{activity_label}"

                    # 写入 Raw 文件 (原封不动)
                    raw_file.write(f"{base_info}," + ",".join(map(str, raw_data)) + "\n")

                    # 计算并写入 Calib 文件
                    calibrated_data = apply_calibration(raw_data, calib_params)
                    calib_file.write(f"{base_info}," + ",".join(map(str, calibrated_data)) + "\n")

                    count += 1
                    if count % 100 == 0:
                        print(f"   📊 已采集 {count} 条数据...", end='\r')
                except ValueError:
                    continue

    except KeyboardInterrupt:
        print("\n\n⏹️ 采集已手动停止。")
    finally:
        ser.close()
        raw_file.close()
        calib_file.close()
        print(f"✅ 完成！共采集 {count} 条数据，两套文件已安全保存。")