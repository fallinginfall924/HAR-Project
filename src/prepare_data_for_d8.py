# D:\HAR-Project\src\check_data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# 获取当前脚本所在的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 原始数据文件路径
csv_path = os.path.join(base_dir, '../data/windowed_dataset.csv')

# --- 修改点 1: 修改输出目录 ---
# 将输出目录直接设为 '../data'，与模型评估代码中的路径保持一致
output_dir = os.path.join(base_dir, '../data')
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. 加载 windowed_dataset.csv ====================
print("🔍 正在检查 windowed_dataset.csv...")
df = pd.read_csv(csv_path)

print(f"原始CSV形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print()

# ==================== 2. 检查数据结构 ====================
# 假设CSV包含: subject_id, label, 以及若干特征列
# 你需要根据实际列名调整下面的代码

# 找出特征列（排除 subject_id 和 label）
exclude_cols = ['subject_id', 'subject', 'label', 'activity', 'Activity']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"特征列数量: {len(feature_cols)}")
print(f"前5个特征列: {feature_cols[:5]}")
print()

# 检查被试信息
if 'subject_id' in df.columns:
    subject_col = 'subject_id'
elif 'subject' in df.columns:
    subject_col = 'subject'
else:
    raise ValueError("找不到被试列 (subject_id 或 subject)")

if 'label' in df.columns:
    label_col = 'label'
elif 'activity' in df.columns:
    label_col = 'activity'
elif 'Activity' in df.columns:
    label_col = 'Activity'
else:
    raise ValueError("找不到标签列 (label/activity/Activity)")

print(f"被试列: {subject_col}")
print(f"标签列: {label_col}")
print()

# 检查每个被试的样本数
print("各被试样本数:")
print(df[subject_col].value_counts().sort_index())
print()

# 检查每个类别的样本数
print("各类别样本数:")
print(df[label_col].value_counts().sort_index())
print()

# ==================== 3. 提取 X, y, groups ====================
X = df[feature_cols].values
y_raw = df[label_col].values
groups = df[subject_col].values

# 标签编码
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"✅ 提取完成:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   groups shape: {groups.shape}")
print(f"   被试数量: {len(np.unique(groups))}")
print(f"   类别数量: {len(le.classes_)}")
print(f"   类别名称: {le.classes_}")
print()

# ==================== 4. 保存 ====================
# --- 修改点 2: 修改输出文件名 ---
# 将文件名修改为模型评估代码所期望的名称
np.save(os.path.join(output_dir, 'X_processed.npy'), X)
np.save(os.path.join(output_dir, 'y_processed.npy'), y)
np.save(os.path.join(output_dir, 'subject_ids.npy'), groups)
joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))

print(f"✅ 数据已保存至 {output_dir}")
print("   - X_processed.npy")
print("   - y_processed.npy")
print("   - subject_ids.npy")
print("   - label_encoder.pkl")