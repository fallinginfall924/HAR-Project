# src/feature_selection.py

import pandas as pd
import numpy as np
import matplotlib
# 【关键修改】强制使用非交互式后端，避免 tkinter 报错和弹窗阻塞
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# 设置路径
DATA_PATH = "../data/feature_matrix.csv"
SAVE_DIR = "../reports/figures"

def load_data():
    """加载数据并清洗"""
    print(f"📂 正在加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 【关键修改】排除 ID 列，防止数据泄露
    if 'subject_id' in df.columns:
        df = df.drop(columns=['subject_id'])
    elif 'id' in df.columns:
        df = df.drop(columns=['id'])

    y = df['label']
    X = df.drop(columns=['label'])
    print(f"✅ 数据加载完成。样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    return X, y

def plot_feature_importance(X, y):
    """1. 特征重要性排名"""
    print("\n🌳 正在计算特征重要性...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importance[indices], align="center", color='#4C72B0')
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "feature_importance_d5.png")
    plt.savefig(save_path, dpi=150)
    plt.close() # 【关键修改】关闭图形释放内存，不再 show
    print(f"✅ 特征重要性图已保存至: {save_path}")

def plot_pca_variance(X):
    """2. PCA 累计方差图"""
    print("\n📊 正在计算 PCA...")
    pca = PCA().fit(X)

    # 计算达到 95% 方差所需的成分数量
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"ℹ️ 统计信息: 前 {n_components_95} 个主成分解释了 95% 的方差。")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumsum)+1), cumsum, marker='o', linestyle='--', color='#DD8452')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
    plt.axvline(x=n_components_95, color='g', linestyle=':', label=f'{n_components_95} Components')
    plt.title('PCA Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "pca_variance_d5.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ PCA 图已保存至: {save_path}")

def plot_tsne(X, y):
    """3. t-SNE 可视化"""
    print("\n🎨 正在计算 t-SNE (可能需要几分钟)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # perplexity 一般取 5-50，样本多可以适当调大
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Activity Label')
    plt.title('t-SNE Visualization of HAR Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(False)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "tsne_d5.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ t-SNE 图已保存至: {save_path}")

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    X, y = load_data()

    # 依次执行三个任务
    plot_feature_importance(X, y)
    plot_pca_variance(X)
    plot_tsne(X, y)

    print("\n🎉 D5 所有可视化任务完成！请查看 reports/figures 目录。")