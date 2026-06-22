# models_advanced.py
# D7: 非线性分类器与集成学习
# 包含：核SVM、kNN、MLP、决策树、随机森林、AdaBoost、GBDT + 超参数调优

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # 用于检查目录
import json

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 非线性分类器
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 集成学习
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（防止画图乱码，如果报错可注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载与预处理 ====================
print("=" * 60)
print("D7: 非线性分类器与集成学习")
print("=" * 60)

# 加载数据
df = pd.read_csv('../data/feature_matrix.csv')

# 1. 剔除 id 列 (关键修复：必须剔除 subject_id，否则 StandardScaler 会报错)
if 'subject_id' in df.columns:
    df = df.drop('subject_id', axis=1)
    print("已剔除 'subject_id' 列")

# 2. 分离特征和标签
X = df.drop('label', axis=1)
y = df['label'].values

# 3. 确保所有特征都是数值型 (双重保险)
X = X.select_dtypes(include=[np.number])

# 转为 numpy 数组
X = X.values

# 建议修改：使用比例划分（例如 20% 作为测试集）
# 假设你总数据约 2430 条，20% 大约是 486 条，这样每个类别就有 80 个左右的样本了
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 标准化（SVM、kNN、MLP 需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"特征维度: {X_train.shape[1]}")
print(f"类别数: {len(np.unique(y))}")
print()

# ==================== 2. 模型定义 ====================
models = {
    # --- 非线性分类器 ---
    'SVM_RBF': SVC(kernel='rbf', random_state=42),
    'SVM_Poly': SVC(kernel='poly', degree=3, random_state=42),
    'kNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),

    # --- 集成学习 ---
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'GBDT': GradientBoostingClassifier(random_state=42),
}

# 需要标准化的模型
need_scaling = ['SVM_RBF', 'SVM_Poly', 'kNN', 'MLP']

# ==================== 3. 基础训练与评估 ====================
print("=" * 60)
print("【基础模型评估】")
print("=" * 60)

results = []

for name, model in models.items():
    # 选择是否使用标准化数据
    if name in need_scaling:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # 训练
    model.fit(X_tr, y_train)

    # 预测
    y_pred = model.predict(X_te)

    # 评估
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    })

    print(f"{name:15} | Test Acc: {acc:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 结果汇总
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print("\n📊 模型对比表:")
print(results_df.to_string(index=False))

# ==================== 4. 超参数调优（以随机森林和SVM为例） ====================
print("\n" + "=" * 60)
print("【超参数调优】")
print("=" * 60)

# --- 随机森林调参 ---
print("\n🔧 随机森林 GridSearchCV...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print(f"最佳参数: {rf_grid.best_params_}")
print(f"最佳CV得分: {rf_grid.best_score_:.4f}")

# --- SVM RBF 调参 ---
print("\n🔧 SVM (RBF) RandomizedSearchCV...")
svm_param_dist = {
    'C': np.logspace(-2, 2, 10),
    'gamma': np.logspace(-3, 1, 10),
    'kernel': ['rbf']
}

svm_search = RandomizedSearchCV(
    SVC(random_state=42),
    svm_param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
svm_search.fit(X_train_scaled, y_train)

print(f"最佳参数: {svm_search.best_params_}")
print(f"最佳CV得分: {svm_search.best_score_:.4f}")

# ==================== 5. 学习曲线分析（以随机森林为例） ====================
print("\n" + "=" * 60)
print("【学习曲线分析】")
print("=" * 60)

# 确保保存目录存在
os.makedirs('../reports/figures', exist_ok=True)

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.title(title)
    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/figures/learning_curve_rf.png', dpi=150)
    plt.show()

# 绘制随机森林学习曲线
plot_learning_curve(
    RandomForestClassifier(**rf_grid.best_params_, random_state=42),
    X_train, y_train,
    'Random Forest - Learning Curve'
)

# ==================== 6. 验证曲线分析（以SVM的gamma为例） ====================
print("\n【验证曲线分析】")

def plot_validation_curve(estimator, X, y, param_name, param_range, title):
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../reports/figures/validation_curve_svm_{param_name}.png', dpi=150)
    plt.show()

# 绘制SVM gamma验证曲线
plot_validation_curve(
    SVC(kernel='rbf', C=1.0, random_state=42),
    X_train_scaled, y_train,
    param_name='gamma',
    param_range=np.logspace(-3, 1, 10),
    title='SVM (RBF) - Validation Curve (gamma)'
)

# ==================== 7. 随机森林特征重要性（反哺D5） ====================
print("\n" + "=" * 60)
print("【特征重要性分析 - 随机森林】")
print("=" * 60)

best_rf = rf_grid.best_estimator_
# 注意：这里重新获取列名，确保和 X 的列对应
feature_names = pd.DataFrame(X_train).columns if isinstance(X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(X_train.shape[1])]
# 如果前面用了 select_dtypes，列名可能会丢失，这里用原始 df 的列名更安全（去掉 subject_id 和 label）
original_feature_names = df.drop('label', axis=1).columns if 'label' in df.columns else df.columns
feature_names = original_feature_names

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n特征重要性排序:")
for i in range(len(feature_names)):
    print(f"  {i + 1:2d}. {feature_names[indices[i]]:20} : {importances[indices[i]]:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
plt.title('Random Forest - Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../reports/figures/feature_importance_rf_d7.png', dpi=150)
plt.show()

# ==================== 8. 最佳模型测试集最终评估 ====================
print("\n" + "=" * 60)
print("【最佳模型最终评估】")
print("=" * 60)

# 使用调优后的随机森林在测试集上评估
final_pred = rf_grid.predict(X_test)
final_acc = accuracy_score(y_test, final_pred)

print(f"\n最佳模型: Random Forest (调优后)")
print(f"测试集准确率: {final_acc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, final_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Random Forest (Tuned)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('../reports/figures/confusion_matrix_rf_tuned.png', dpi=150)
plt.show()

# ==================== 9. 保存调参记录 ====================
tuning_log = {
    'RandomForest_Best_Params': rf_grid.best_params_,
    'RandomForest_Best_CV_Score': rf_grid.best_score_,
    'SVM_RBF_Best_Params': svm_search.best_params_,
    'SVM_RBF_Best_CV_Score': svm_search.best_score_,
    'Final_Test_Accuracy': final_acc
}

with open('../reports/tuning_log_d7.json', 'w') as f:
    json.dump(tuning_log, f, indent=2, ensure_ascii=False)

print("\n✅ 调参记录已保存至: ../reports/tuning_log_d7.json")
print("✅ 所有图表已保存至: ../reports/figures/")
print("\n🎉 D7 任务完成！")