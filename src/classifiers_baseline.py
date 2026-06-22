# -*- coding: utf-8 -*-
"""
D6: 基础统计分类器基线 (最终修正版)
修复点：
1. 移除已废弃的 multi_class 参数
2. 彻底重写最小风险决策模块：基于 GNB 先验概率调整，解决维度报错并提升召回率
3. 优化 LogReg 参数以匹配高准确率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# --- 配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = "../data/feature_matrix.csv"
SAVE_DIR = "../reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 数据加载 ====================
print("📂 加载数据...")
df = pd.read_csv(DATA_PATH)

# 剔除 ID 列
drop_cols = [c for c in df.columns if 'id' in c.lower()]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"   - 已剔除列: {drop_cols}")

y = df['label'].values
X = df.drop('label', axis=1).values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]} 样本 | 测试集: {X_test.shape[0]} 样本 | 维度: {X_train.shape[1]}")

# ==================== 模型训练与评估 ====================
results = {}

# 1. 高斯贝叶斯
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
results['GNB'] = {
    'acc': accuracy_score(y_test, y_pred_gnb),
    'recall': recall_score(y_test, y_pred_gnb, average='macro'),
    'pred': y_pred_gnb
}

# 2. LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
results['LDA'] = {
    'acc': accuracy_score(y_test, y_pred_lda),
    'recall': recall_score(y_test, y_pred_lda, average='macro'),
    'pred': y_pred_lda
}

# 3. 逻辑回归 (Tuned)
logreg = LogisticRegression(max_iter=2000, C=10.0, random_state=42, solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
results['LogReg'] = {
    'acc': accuracy_score(y_test, y_pred_logreg),
    'recall': recall_score(y_test, y_pred_logreg, average='macro'),
    'pred': y_pred_logreg
}

# 4. 线性 SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
results['SVM'] = {
    'acc': accuracy_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm, average='macro'),
    'pred': y_pred_svm
}

# ==================== 结果汇总 ====================
print("\n" + "=" * 50)
print("📊 基线模型对比汇总")
print("=" * 50)
summary = pd.DataFrame({
    '模型': list(results.keys()),
    '准确率': [v['acc'] for v in results.values()],
    '宏平均召回率': [v['recall'] for v in results.values()]
})
print(summary.to_string(index=False))
summary.to_csv(os.path.join(SAVE_DIR, "baseline_comparison.csv"), index=False)

# ==================== 混淆矩阵可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, res) in zip(axes.flat, results.items()):
    cm = confusion_matrix(y_test, res['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(f'{name} - 混淆矩阵 (Acc: {res["acc"]:.4f})')
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrices.png"), dpi=150)
plt.show()

# ==================== 最小风险贝叶斯决策 (核心修复) ====================
print("\n" + "=" * 50)
print("⚠️ 最小风险贝叶斯决策 (Cost-Sensitive)")
print("=" * 50)

FALL_LABEL = 5  # 假设跌倒为类别5，请根据实际情况修改
COST_FACTOR = 10  # 漏检代价倍数

# 1. 获取原始先验概率 P(y)
original_priors = gnb.class_prior_.copy()

# 2. 调整先验概率：模拟“漏检代价高”
# 原理：提高某类的先验概率，等价于降低该类的误判代价
risk_priors = original_priors.copy()
fall_idx = np.where(gnb.classes_ == FALL_LABEL)[0]

if len(fall_idx) > 0:
    fall_idx = fall_idx[0]
    risk_priors[fall_idx] *= COST_FACTOR
    # 归一化，保证概率和为1
    risk_priors /= risk_priors.sum()

    # 3. 使用调整后的先验重新计算后验概率并预测
    # GNB 的后验概率公式：P(y|x) ∝ P(x|y) * P(y)
    # sklearn 的 predict_log_proba 返回的是 log(P(x|y)) + log(P(y_original))
    # 我们需要减去旧先验，加上新先验
    log_likelihoods = gnb.predict_log_proba(X_test)
    log_old_priors = np.log(original_priors)
    log_new_priors = np.log(risk_priors)

    # 修正后的对数后验
    log_posterior_risk = log_likelihoods - log_old_priors + log_new_priors

    y_pred_risk = np.argmax(log_posterior_risk, axis=1)

    # 4. 评估
    acc_default = results['GNB']['acc']
    acc_risk = accuracy_score(y_test, y_pred_risk)

    rec_default = recall_score(y_test, results['GNB']['pred'], labels=[FALL_LABEL], average=None)[0]
    rec_risk = recall_score(y_test, y_pred_risk, labels=[FALL_LABEL], average=None)[0]

    print(f"【默认决策】跌倒召回率: {rec_default:.4f} | 整体准确率: {acc_default:.4f}")
    print(f"【最小风险】跌倒召回率: {rec_risk:.4f} | 整体准确率: {acc_risk:.4f}")
    print(f"💡 变化: 召回率 {'↑' if rec_risk > rec_default else '↓'} {(rec_risk - rec_default) * 100:+.2f}%")
else:
    print(f"❌ 未找到标签 {FALL_LABEL}，跳过最小风险决策。")

# ==================== 决策边界可视化 (PCA 2D) ====================
print("\n🎨 生成决策边界图 (PCA 2D)...")
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
models_2d = {'GNB': GaussianNB(), 'LDA': LinearDiscriminantAnalysis(),
             'LogReg': LogisticRegression(max_iter=2000, C=10.0), 'SVM': SVC(kernel='linear')}

x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, (name, model) in zip(axes.flat, models_2d.items()):
    model.fit(X_train_2d, y_train)
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap='viridis', edgecolors='k', s=20)
    ax.set_title(f'{name} Decision Boundary (PCA)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "decision_boundaries.png"), dpi=150)
plt.show()

print("\n🎉 D6 任务完成！结果已保存。")