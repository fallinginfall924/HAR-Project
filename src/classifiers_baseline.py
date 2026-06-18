# -*- coding: utf-8 -*-
"""
D6: 基础统计分类器基线
涵盖：高斯贝叶斯、朴素贝叶斯、LDA、逻辑回归、线性SVM、最小风险决策、决策边界可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
# --- 修复：添加中文字体支持 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# ---------------------------
# ==================== 配置 ====================
DATA_PATH = "../data/feature_matrix.csv"
SAVE_DIR = "../reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 数据加载 ====================
print("📂 加载数据...")
df = pd.read_csv(DATA_PATH)

# 剔除 id 列（如果存在）
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

y = df['label'].values
X = df.drop('label', axis=1).values

# 标准化（LDA、逻辑回归、SVM 需要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
print(f"特征维度: {X_train.shape[1]}")

# ==================== 1. 高斯贝叶斯分类器 ====================
print("\n" + "=" * 50)
print("1. 高斯贝叶斯分类器 (Gaussian Naive Bayes)")
print("=" * 50)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, y_pred_gnb)
recall_gnb = recall_score(y_test, y_pred_gnb, average='macro')

print(f"准确率: {acc_gnb:.4f}")
print(f"宏平均召回率: {recall_gnb:.4f}")

# ==================== 2. 线性判别分析 (LDA) ====================
print("\n" + "=" * 50)
print("2. 线性判别分析 (Fisher LDA)")
print("=" * 50)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
acc_lda = accuracy_score(y_test, y_pred_lda)
recall_lda = recall_score(y_test, y_pred_lda, average='macro')

print(f"准确率: {acc_lda:.4f}")
print(f"宏平均召回率: {recall_lda:.4f}")

# ==================== 3. 逻辑回归 ====================
print("\n" + "=" * 50)
print("3. 逻辑回归 (Logistic Regression)")
print("=" * 50)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg, average='macro')

print(f"准确率: {acc_logreg:.4f}")
print(f"宏平均召回率: {recall_logreg:.4f}")

# ==================== 4. 线性 SVM ====================
print("\n" + "=" * 50)
print("4. 线性 SVM (Linear SVM)")
print("=" * 50)

svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_svm = svm_linear.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm, average='macro')

print(f"准确率: {acc_svm:.4f}")
print(f"宏平均召回率: {recall_svm:.4f}")

# ==================== 5. 基线精度对比表 ====================
print("\n" + "=" * 50)
print("📊 基线模型对比汇总")
print("=" * 50)

results = pd.DataFrame({
    '模型': ['高斯贝叶斯', 'LDA', '逻辑回归', '线性SVM'],
    '准确率': [acc_gnb, acc_lda, acc_logreg, acc_svm],
    '宏平均召回率': [recall_gnb, recall_lda, recall_logreg, recall_svm]
})
print(results.to_string(index=False))

# 保存对比表
results.to_csv(os.path.join(SAVE_DIR, "baseline_accuracy_comparison.csv"), index=False)

# ==================== 6. 混淆矩阵可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
models = [
    ('高斯贝叶斯', y_pred_gnb),
    ('LDA', y_pred_lda),
    ('逻辑回归', y_pred_logreg),
    ('线性SVM', y_pred_svm)
]

for ax, (name, y_pred) in zip(axes.flat, models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} - 混淆矩阵\n准确率: {accuracy_score(y_test, y_pred):.4f}')
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrices_baseline.png"), dpi=150, bbox_inches='tight')
plt.show()

# ==================== 7. 最小风险贝叶斯决策（代价敏感） ====================
print("\n" + "=" * 50)
print("7. 最小风险贝叶斯决策（代价敏感学习）")
print("=" * 50)

# 假设：类别 5 是"跌倒"（请根据你的实际数据调整）
FALL_CLASS = 5

# 设计代价矩阵：漏检跌倒（将跌倒预测为其他类）的代价极高
# 代价矩阵 C[i,j] = 将真实类别 i 预测为类别 j 的代价
num_classes = len(np.unique(y))
cost_matrix = np.ones((num_classes, num_classes))  # 默认代价为1
np.fill_diagonal(cost_matrix, 0)  # 正确分类代价为0

# 跌倒漏检的代价设为10（远高于其他错误）
for j in range(num_classes):
    if j != FALL_CLASS:
        cost_matrix[FALL_CLASS, j] = 10.0

print("代价矩阵（行=真实类别，列=预测类别）:")
print(cost_matrix)

# 使用高斯贝叶斯的概率预测来实现最小风险决策
y_proba = gnb.predict_proba(X_test)

# 计算每个样本的期望风险（对每个可能的预测类别）
expected_risk = np.dot(y_proba, cost_matrix.T)  # shape: (n_samples, n_classes)

# 最小风险决策：选择期望风险最小的类别
y_pred_risk = np.argmin(expected_risk, axis=1)

acc_risk = accuracy_score(y_test, y_pred_risk)
recall_fall_risk = recall_score(y_test, y_pred_risk, labels=[FALL_CLASS], average=None)[0]
recall_fall_default = recall_score(y_test, y_pred_gnb, labels=[FALL_CLASS], average=None)[0]

print(f"\n【最小错误率决策】跌倒类(类别{FALL_CLASS})召回率: {recall_fall_default:.4f}")
print(f"【最小风险决策】跌倒类(类别{FALL_CLASS})召回率: {recall_fall_risk:.4f}")
print(f"整体准确率变化: {acc_gnb:.4f} → {acc_risk:.4f}")

# 保存代价敏感对比
risk_comparison = pd.DataFrame({
    '决策策略': ['最小错误率', '最小风险（代价敏感）'],
    '跌倒类召回率': [recall_fall_default, recall_fall_risk],
    '整体准确率': [acc_gnb, acc_risk]
})
risk_comparison.to_csv(os.path.join(SAVE_DIR, "risk_decision_comparison.csv"), index=False)

# ==================== 8. 决策边界可视化（PCA 2D 投影） ====================
print("\n" + "=" * 50)
print("8. 决策边界可视化（PCA 2D 投影）")
print("=" * 50)

from sklearn.decomposition import PCA

# 将数据降到2D用于可视化
pca_2d = PCA(n_components=2)
X_train_2d = pca_2d.fit_transform(X_train)
X_test_2d = pca_2d.transform(X_test)

# 在2D空间重新训练各模型
models_2d = {
    '高斯贝叶斯': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '线性SVM': SVC(kernel='linear', random_state=42)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flat

# 生成网格用于绘制决策边界
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, (name, model) in zip(axes, models_2d.items()):
    # 在2D数据上训练
    model.fit(X_train_2d, y_train)

    # 预测网格点
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(grid)
        # 绘制概率热力图（取最大概率类别）
        Z_class = np.argmax(Z, axis=1)
    else:
        Z_class = model.predict(grid)

    Z_class = Z_class.reshape(xx.shape)

    # 绘制决策边界
    contour = ax.contourf(xx, yy, Z_class, alpha=0.4, cmap='viridis')

    # 绘制训练数据点
    scatter = ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train,
                         cmap='viridis', edgecolors='k', s=50)

    ax.set_title(f'{name} - 决策边界 (PCA 2D)', fontsize=12)
    ax.set_xlabel('第一主成分')
    ax.set_ylabel('第二主成分')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "decision_boundaries_2d.png"), dpi=150, bbox_inches='tight')
plt.show()

print("\n🎉 D6 所有任务完成！")
print(f"📁 结果已保存至: {SAVE_DIR}")