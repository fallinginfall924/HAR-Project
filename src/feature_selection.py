# src/feature_selection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# 设置路径
DATA_PATH = "../data/feature_matrix.csv"
MODEL_PATH = "../models/har_model.pkl" # 假设 train.py 保存了模型
SAVE_DIR = "../reports/figures"

def load_data():
    """加载数据"""
    df = pd.read_csv(DATA_PATH)
    y = df['label']
    X = df.drop('label', axis=1)
    return X, y

def plot_feature_importance(X, y):
    """
    1. 特征重要性排名 (Embedded方法)
    这里我们现场训练一个RF来获取重要性，或者加载已保存的模型
    """
    print("🌳 正在计算特征重要性...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importance[indices], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "feature_importance_d5.png"))
    plt.show()
    print("✅ 特征重要性图已保存")

def plot_pca_variance(X):
    """
    2. PCA 累计方差图
    """
    print("📊 正在计算 PCA...")
    pca = PCA().fit(X)

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    plt.title('PCA Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "pca_variance_d5.png"))
    plt.show()
    print("✅ PCA 图已保存")

def plot_tsne(X, y):
    """
    3. t-SNE 可视化
    """
    print("🎨 正在计算 t-SNE...")
    # t-SNE 对尺度敏感，必须标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Activity Label')
    plt.title('t-SNE Visualization of HAR Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(os.path.join(SAVE_DIR, "tsne_d5.png"))
    plt.show()
    print("✅ t-SNE 图已保存")

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    X, y = load_data()

    # 依次执行三个任务
    plot_feature_importance(X, y)
    plot_pca_variance(X)
    plot_tsne(X, y)

    print("\n🎉 D5 所有可视化任务完成！请查看 reports/figures 目录。")