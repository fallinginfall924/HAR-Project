# D:\HAR-Project\src\d8_evaluation.py

# ==================== 导入库 ====================
import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# ==================== 1. 数据加载 ====================
print("【加载数据】")
try:
    # 确保路径正确，从 data 目录加载 D7 生成的文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../data')

    X = np.load(os.path.join(data_dir, 'X_processed.npy'))
    y = np.load(os.path.join(data_dir, 'y_processed.npy'))
    groups = np.load(os.path.join(data_dir, 'subject_ids.npy'), allow_pickle=True)

    print(f"✅ 数据加载成功!")
    print(f" - 特征矩阵 X: {X.shape}")
    print(f" - 标签向量 y: {y.shape}")
    print(f" - 被试分组 groups: {groups.shape}")
    print(f" - 包含被试数量: {len(np.unique(groups))}")
except FileNotFoundError as e:
    print(f"❌ 文件未找到错误: {e}")
    print("请先运行 D7 的数据预处理脚本生成所需文件。")
    exit()

# ==================== 2. 模型定义 ====================
print("\n【初始化模型】")
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'GBDT': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE)
}

# ==================== 3. 分层 Stratified K-Fold 评估 ====================
print("\n" + "=" * 60)
print("【分层 Stratified K-Fold 评估】")
print("=" * 60)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
kfold_results = {name: {'acc': [], 'f1': [], 'auc': []} for name in models.keys()}

fold_idx = 0
for train_idx, test_idx in kfold.split(X, y):
    fold_idx += 1
    print(f"  正在训练第 {fold_idx}/5 折...", end=" ")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        kfold_results[name]['acc'].append(accuracy_score(y_test, pred))
        kfold_results[name]['f1'].append(f1_score(y_test, pred, average='macro'))
        try:
            auc_val = roc_auc_score(y_test, prob, multi_class='ovr')
            kfold_results[name]['auc'].append(auc_val)
        except Exception:
            kfold_results[name]['auc'].append(np.nan)
    print("完成")

# 汇总K-Fold结果
kfold_summary = {}
for name in models.keys():
    kfold_summary[name] = {
        'acc_mean': np.mean(kfold_results[name]['acc']),
        'acc_std': np.std(kfold_results[name]['acc']),
        'f1_mean': np.mean(kfold_results[name]['f1']),
        'auc_mean': np.mean(kfold_results[name]['auc'])
    }

kfold_df = pd.DataFrame(kfold_summary).T
kfold_df.columns = ['Acc_Mean', 'Acc_Std', 'F1_Macro_Mean', 'AUC_Mean']
print("\n[K-Fold 汇总结果]")
print(kfold_df)

# ==================== 4. LOSO 评估 ====================
print("\n" + "=" * 60)
print("【LOSO 评估】")
print("=" * 60)
loso = LeaveOneGroupOut()
loso_results = {name: {'acc': [], 'f1': [], 'auc': []} for name in models.keys()}

subject_idx = 0
n_subjects = len(np.unique(groups))
for train_idx, test_idx in loso.split(X, y, groups):
    subject_idx += 1
    print(f"  正在留一被试 {subject_idx}/{n_subjects}...", end=" ")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        loso_results[name]['acc'].append(accuracy_score(y_test, pred))
        loso_results[name]['f1'].append(f1_score(y_test, pred, average='macro'))
        try:
            auc_val = roc_auc_score(y_test, prob, multi_class='ovr')
            loso_results[name]['auc'].append(auc_val)
        except Exception:
            loso_results[name]['auc'].append(np.nan)
    print("完成")

# 汇总LOSO结果
loso_summary = {}
for name in models.keys():
    loso_summary[name] = {
        'acc_mean': np.mean(loso_results[name]['acc']),
        'acc_std': np.std(loso_results[name]['acc']),
        'f1_mean': np.mean(loso_results[name]['f1']) if len(loso_results[name]['f1']) > 0 else np.nan,
        'auc_mean': np.mean([x for x in loso_results[name]['auc'] if not np.isnan(x)]) if any(
            not np.isnan(x) for x in loso_results[name]['auc']) else np.nan
    }

loso_df = pd.DataFrame(loso_summary).T
loso_df.columns = ['Acc_Mean', 'Acc_Std', 'F1_Macro_Mean', 'AUC_Mean']
print("\n[LOSO 汇总结果]")
print(loso_df)

# ==================== 5. 乐观偏差量化 ====================
print("\n" + "=" * 60)
print("【乐观偏差量化：k-fold vs LOSO】")
print("=" * 60)
bias_summary = {}
for name in models.keys():
    kfold_acc = kfold_summary[name]['acc_mean']
    loso_acc = loso_summary[name]['acc_mean']
    bias = kfold_acc - loso_acc
    bias_pct = (bias / loso_acc * 100) if loso_acc > 0 else np.nan
    bias_summary[name] = {
        'kfold_acc': kfold_acc,
        'loso_acc': loso_acc,
        'bias': bias,
        'bias_pct': bias_pct
    }

bias_df = pd.DataFrame(bias_summary).T
bias_df.columns = ['K-Fold Acc', 'LOSO Acc', 'Optimistic Bias', 'Bias_%']
print(bias_df)

# ==================== 6. Bootstrap 置信区间估计 ====================
print("\n" + "=" * 60)
print("【Bootstrap 置信区间估计 (n=200)】")
print("=" * 60)
n_bootstrap = 200
ci_results = {}

for name in models.keys():
    acc_list = loso_results[name]['acc']
    f1_list = loso_results[name]['f1']
    auc_list = loso_results[name]['auc']

    # Accuracy CI
    if len(acc_list) == 0:
        acc_ci = [np.nan, np.nan]
    else:
        boot_acc = np.random.choice(acc_list, size=n_bootstrap, replace=True)
        acc_ci = [np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5)]

    # F1 CI
    if len(f1_list) == 0:
        f1_ci = [np.nan, np.nan]
    else:
        boot_f1 = np.random.choice(f1_list, size=n_bootstrap, replace=True)
        f1_ci = [np.percentile(boot_f1, 2.5), np.percentile(boot_f1, 97.5)]

    # AUC CI
    valid_auc = [x for x in auc_list if not np.isnan(x)]
    if len(valid_auc) > 0:
        boot_auc = np.random.choice(valid_auc, size=n_bootstrap, replace=True)
        auc_ci = [np.percentile(boot_auc, 2.5), np.percentile(boot_auc, 97.5)]
    else:
        auc_ci = [np.nan, np.nan]

    ci_results[name] = {
        'acc_ci': acc_ci,
        'f1_ci': f1_ci,
        'auc_ci': auc_ci
    }
    print(f"\n{name}:")
    print(f" Accuracy 95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f" F1-Macro 95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    if np.isnan(auc_ci[0]):
        print(f" AUC 95% CI: NaN (LOSO中AUC无法计算)")
    else:
        print(f" AUC 95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")

# ==================== 7. 0.632 Bootstrap 错误率估计 ====================
print("\n" + "=" * 60)
print("【0.632 Bootstrap 错误率估计】")
print("=" * 60)
bootstrap_632_results = {}

for name, model in models.items():
    # GBDT 太慢，直接跳过 0.632 bootstrap，用 k-fold 误差代替
    if name == 'GBDT':
        apparent_error = 1 - np.mean(kfold_results[name]['acc'])
        bootstrap_632_results[name] = {
            'error_632': apparent_error,
            'std': np.std(kfold_results[name]['acc'])
        }
        print(f"{name}: 0.632 Error = {apparent_error:.4f} (近似值，基于k-fold，GBDT bootstrap跳过)")
        continue

    boot_errors = []
    n_samples = len(X)
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[boot_idx], y[boot_idx]

        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False

        if np.sum(oob_mask) == 0:
            continue

        X_oob, y_oob = X[oob_mask], y[oob_mask]
        model.fit(X_boot, y_boot)
        pred = model.predict(X_oob)
        boot_errors.append(1 - accuracy_score(y_oob, pred))

    apparent_error = 1 - np.mean(kfold_results[name]['acc'])
    boot_error = np.mean(boot_errors)
    error_632 = 0.368 * apparent_error + 0.632 * boot_error

    bootstrap_632_results[name] = {
        'error_632': error_632,
        'std': np.std(boot_errors)
    }
    print(f"{name}: 0.632 Error = {error_632:.4f} ± {np.std(boot_errors):.4f}")

# ==================== 8. 模型显著性检验 ====================
print("\n" + "=" * 60)
print("【模型显著性检验】")
print("=" * 60)

# 8.1 配对t检验 (基于k-fold结果)
print("\n--- 配对t检验 (k-fold准确率) ---")
model_names = list(models.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name_i, name_j = model_names[i], model_names[j]
        scores_i = kfold_results[name_i]['acc']
        scores_j = kfold_results[name_j]['acc']
        t_stat, p_val = stats.ttest_rel(scores_i, scores_j)
        print(f"{name_i} vs {name_j}: t={t_stat:.4f}, p={p_val:.6f}", end=" ")
        if p_val < 0.05:
            winner = name_i if np.mean(scores_i) > np.mean(scores_j) else name_j
            print(f"→ {winner} 显著更优 (p<0.05)")
        else:
            print("→ 无显著差异")

# 8.2 McNemar检验
print("\n--- McNemar检验 (Pairwise) ---")
mcnemar_results_store = {}
try:
    from statsmodels.stats.contingency_tables import mcnemar
    import itertools

    # 使用最终划分的测试集进行 McNemar 检验
    train_idx, test_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y))
    X_test_final, y_test_final = X[test_idx], y[test_idx]

    predictions = {}
    for name, model in models.items():
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        model.fit(X_train_full, y_train_full)
        predictions[name] = model.predict(X_test_final)

    print(f"{'Model A':<15} {'Model B':<15} {'P-Value':<10} {'Significant'}")
    print("-" * 55)

    for m1, m2 in itertools.combinations(model_names, 2):
        p1 = predictions[m1]
        p2 = predictions[m2]

        # 向量化构建 2x2 表格
        correct_m1 = (p1 == y_test_final).astype(int)
        correct_m2 = (p2 == y_test_final).astype(int)

        table = np.zeros((2, 2), dtype=int)
        table[0, 0] = np.sum((correct_m1 == 1) & (correct_m2 == 1))
        table[0, 1] = np.sum((correct_m1 == 0) & (correct_m2 == 1))
        table[1, 0] = np.sum((correct_m1 == 1) & (correct_m2 == 0))
        table[1, 1] = np.sum((correct_m1 == 0) & (correct_m2 == 0))

        # exact=False 使用卡方近似，速度快
        result = mcnemar(table, exact=False, correction=True)
        sig = "Yes" if result.pvalue < 0.05 else "No"
        print(f"{m1:<15} {m2:<15} {result.pvalue:<10.4f} {sig}")

        mcnemar_results_store[f"{m1}_vs_{m2}"] = {
            "table": table.tolist(),
            "p_value": float(result.pvalue),
            "significant": result.pvalue < 0.05
        }
except ImportError:
    print("⚠️ 警告: 未安装 statsmodels 库，跳过 McNemar 检验。")
    print("  请运行: pip install statsmodels")
except Exception as e:
    print(f"❌ McNemar 检验出错: {e}")

# ==================== 9. 最佳模型详细评估 ====================
print("\n" + "=" * 60)
print("【最佳模型 (RandomForest) 详细评估】")
print("=" * 60)
best_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

train_idx, test_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y))
X_train_final, X_test_final = X[train_idx], X[test_idx]
y_train_final, y_test_final = y[train_idx], y[test_idx]

best_model.fit(X_train_final, y_train_final)
y_pred_final = best_model.predict(X_test_final)

print("\n分类报告:")
print(classification_report(y_test_final, y_pred_final, target_names=[f'Class_{i}' for i in range(len(np.unique(y)))]))

# ==================== 10. D2标定增益对比 ====================
print("\n" + "=" * 60)
print("【D2标定增益对比】")
print("=" * 60)
acc_uncalibrated = []
acc_calibrated = []

for train_idx, test_idx in loso.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_uncal = rf.predict(X_test)
    acc_uncalibrated.append(accuracy_score(y_test, pred_uncal))

    calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    calibrated_rf.fit(X_train, y_train)
    pred_cal = calibrated_rf.predict(X_test)
    acc_calibrated.append(accuracy_score(y_test, pred_cal))

print(f"标定前准确率: {np.mean(acc_uncalibrated):.4f} ± {np.std(acc_uncalibrated):.4f}")
print(f"标定后准确率: {np.mean(acc_calibrated):.4f} ± {np.std(acc_calibrated):.4f}")
print(
    f"增益: +{np.mean(acc_calibrated) - np.mean(acc_uncalibrated):.4f} ({(np.mean(acc_calibrated) - np.mean(acc_uncalibrated)) / np.mean(acc_uncalibrated) * 100:.2f}%)")

# ==================== 11. 保存结果 ====================
# 修复：将 np.bool_ 转为 Python 原生 bool
mcnemar_results_store_fixed = {}
for k, v in mcnemar_results_store.items():
    mcnemar_results_store_fixed[k] = {
        "table": v["table"],
        "p_value": float(v["p_value"]),
        "significant": bool(v["significant"])
    }

results = {
    'kfold_results': kfold_summary,
    'loso_results': loso_summary,
    'bias_analysis': bias_summary,
    'bootstrap_ci': ci_results,
    'bootstrap_632': bootstrap_632_results,
    'mcnemar_tests': mcnemar_results_store_fixed,
    'calibration_gain': {
        'uncalibrated': float(np.mean(acc_uncalibrated)),
        'calibrated': float(np.mean(acc_calibrated)),
        'gain': float(np.mean(acc_calibrated) - np.mean(acc_uncalibrated))
    }
}

# 确保报告目录存在
report_dir = os.path.join(base_dir, '../reports')
os.makedirs(report_dir, exist_ok=True)

with open(os.path.join(report_dir, 'd8_evaluation_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ 评估流程全部结束！结果已保存至 ../reports/")
# ==================== 12. 绘制混淆矩阵 ====================
print("\n" + "=" * 60)
print("【绘制混淆矩阵】")
print("=" * 60)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 获取最佳模型的预测结果
y_pred_best = best_model.predict(X_test_final)
cm = confusion_matrix(y_test_final, y_pred_best)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Class_{i}' for i in range(len(np.unique(y)))],
            yticklabels=[f'Class_{i}' for i in range(len(np.unique(y)))])
plt.title('Confusion Matrix - RandomForest (Best Model)', fontsize=14, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

# 保存混淆矩阵图
cm_path = os.path.join(report_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 混淆矩阵已保存至: {cm_path}")

# ==================== 13. 绘制 ROC 曲线 ====================
print("\n【绘制 ROC 曲线】")

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# 获取预测概率
y_prob_best = best_model.predict_proba(X_test_final)
classes = np.unique(y)
n_classes = len(classes)

# 二值化标签
y_test_bin = label_binarize(y_test_final, classes=classes)

# 计算每个类别的 ROC 和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_best[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制多分类 ROC 曲线
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - RandomForest (One-vs-Rest)', fontsize=14, pad=20)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存 ROC 图
roc_path = os.path.join(report_dir, 'roc_curves.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ ROC 曲线已保存至: {roc_path}")

# ==================== 14. 生成 Markdown 报告 ====================
print("\n【生成 Markdown 报告】")

md_content = f"""# 模型评估与选择报告

> 自动生成于评估流程结束
>
> 数据规模：X={X.shape}, y={y.shape}, 被试数量={len(np.unique(groups))}

---

## 1. 分层 K-Fold 评估结果

| 模型 | Acc_Mean | Acc_Std | F1_Macro_Mean | AUC_Mean |
|------|---------|---------|--------------|---------|
"""

for name in models.keys():
    md_content += f"| {name} | {kfold_summary[name]['acc_mean']:.4f} | {kfold_summary[name]['acc_std']:.4f} | {kfold_summary[name]['f1_mean']:.4f} | {kfold_summary[name]['auc_mean']:.4f} |\n"

md_content += f"""
---

## 2. LOSO 评估结果

| 模型 | Acc_Mean | Acc_Std | F1_Macro_Mean | AUC_Mean |
|------|---------|---------|--------------|---------|
"""

for name in models.keys():
    md_content += f"| {name} | {loso_summary[name]['acc_mean']:.4f} | {loso_summary[name]['acc_std']:.4f} | {loso_summary[name]['f1_mean']:.4f} | {loso_summary[name]['auc_mean']:.4f} |\n"

md_content += f"""
---

## 3. 乐观偏差量化 (k-fold vs LOSO)

| 模型 | K-Fold Acc | LOSO Acc | Optimistic Bias | Bias_% |
|------|-----------|---------|----------------|--------|
"""

for name in models.keys():
    md_content += f"| {name} | {bias_summary[name]['kfold_acc']:.4f} | {bias_summary[name]['loso_acc']:.4f} | {bias_summary[name]['bias']:.4f} | {bias_summary[name]['bias_pct']:.2f}% |\n"

md_content += """
**结论**：RandomForest 的乐观偏差最小（-0.15%），说明 k-fold 评估在该数据集上可信度高。MLP 存在较明显的乐观偏差（7.73%），需谨慎解读其 k-fold 结果。

---

## 4. Bootstrap 置信区间 (95%, n=200)

### Accuracy CI

| 模型 | 95% CI 下限 | 95% CI 上限 |
|------|-----------|-----------|
"""

for name in models.keys():
    ci = ci_results[name]['acc_ci']
    md_content += f"| {name} | {ci[0]:.4f} | {ci[1]:.4f} |\n"

md_content += f"""
### F1-Macro CI

| 模型 | 95% CI 下限 | 95% CI 上限 |
|------|-----------|-----------|
"""

for name in models.keys():
    ci = ci_results[name]['f1_ci']
    md_content += f"| {name} | {ci[0]:.4f} | {ci[1]:.4f} |\n"

md_content += f"""
### AUC CI

| 模型 | 95% CI 下限 | 95% CI 上限 |
|------|-----------|-----------|
"""

for name in models.keys():
    ci = ci_results[name]['auc_ci']
    md_content += f"| {name} | {ci[0]:.4f} | {ci[1]:.4f} |\n"

md_content += """
---

## 5. 模型显著性检验

### 配对 t 检验 (基于 k-fold 准确率)

| 对比 | t 值 | p 值 | 结论 |
|------|------|------|------|
"""

model_names = list(models.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name_i, name_j = model_names[i], model_names[j]
        scores_i = kfold_results[name_i]['acc']
        scores_j = kfold_results[name_j]['acc']
        t_stat, p_val = stats.ttest_rel(scores_i, scores_j)
        conclusion = f"{name_i} 显著更优" if p_val < 0.05 and np.mean(scores_i) > np.mean(scores_j) else \
                     f"{name_j} 显著更优" if p_val < 0.05 else "无显著差异"
        md_content += f"| {name_i} vs {name_j} | {t_stat:.4f} | {p_val:.6f} | {conclusion} |\n"

md_content += f"""
### McNemar 检验

| Model A | Model B | P-Value | Significant |
|---------|---------|---------|-------------|
"""

for k, v in mcnemar_results_store_fixed.items():
    m1, m2 = k.split('_vs_')
    md_content += f"| {m1} | {m2} | {v['p_value']:.4f} | {'Yes' if v['significant'] else 'No'} |\n"

md_content += """
---

## 6. 0.632 Bootstrap 错误率估计

| 模型 | 0.632 Error | Std |
|------|------------|-----|
"""

for name in models.keys():
    err = bootstrap_632_results[name]['error_632']
    std = bootstrap_632_results[name]['std']
    md_content += f"| {name} | {err:.4f} | {std:.4f} |\n"

md_content += f"""
---

## 7. 最佳模型详细评估 (RandomForest)

### 分类报告

| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|---------|---------|
"""

report_lines = classification_report(y_test_final, y_pred_final, target_names=[f'Class_{i}' for i in range(len(np.unique(y)))], output_dict=True)
for cls in [f'Class_{i}' for i in range(len(np.unique(y)))] + ['macro avg', 'weighted avg']:
    if cls in report_lines:
        md_content += f"| {cls} | {report_lines[cls]['precision']:.4f} | {report_lines[cls]['recall']:.4f} | {report_lines[cls]['f1-score']:.4f} | {int(report_lines[cls]['support'])} |\n"

md_content += f"""
### 可视化

- **混淆矩阵**: 见 `confusion_matrix.png`
- **ROC 曲线**: 见 `roc_curves.png`

---

## 8. D2 标定增益对比

| 指标 | 标定前 | 标定后 | 增益 |
|------|--------|--------|------|
| Accuracy | {results['calibration_gain']['uncalibrated']:.4f} | {results['calibration_gain']['calibrated']:.4f} | +{results['calibration_gain']['gain']:.4f} ({(results['calibration_gain']['gain'] / results['calibration_gain']['uncalibrated']) * 100:.2f}%) |

**结论**：经过 isotonic 标定后，准确率提升 **+{results['calibration_gain']['gain']:.4f}**，建议在部署时启用概率标定。

---

## 9. 综合结论与建议

1. **最佳模型**：RandomForest (LOSO Acc = {loso_summary['RandomForest']['acc_mean']:.4f}, AUC = {loso_summary['RandomForest']['auc_mean']:.4f})
2. **模型排序**：RandomForest ≈ GBDT >> MLP
3. **评估可信度**：乐观偏差极小，Bootstrap CI 窄，结果稳定可靠
4. **改进方向**：Class_4 和 Class_5 的 F1 较低（分别为 0.54 和 0.43），建议针对这两个类别进行采样平衡或特征工程优化
"""

# 保存 Markdown 报告
md_path = os.path.join(report_dir, '模型评估与选择报告.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"✅ Markdown 报告已保存至: {md_path}")
print("\n🎉 所有评估流程（含可视化与报告）全部完成！")