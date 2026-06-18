import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# --- 配置 ---
DATA_PATH = "../data/feature_matrix.csv"
MODEL_SAVE_PATH = "../models/har_model.pkl"
PLOT_SAVE_DIR = "../reports/figures"  # 用于保存混淆矩阵和特征重要性图

# 确保保存图表的目录存在
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


def load_and_prepare_data(path):
    """加载数据并分离特征与标签"""
    print(f"📂 正在加载数据: {path}")
    df = pd.read_csv(path)

    # 假设 'id' 和 'label' 是前两列，其余为特征
    # 请根据你的 feature_matrix.csv 实际列名进行调整
    # 这里假设列名为 'id', 'label', 'mean_x', 'std_mag', ...
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    y = df['label']
    X = df.drop(columns=['label'])

    print(f"✅ 数据加载完成。样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    return X, y


def train_and_evaluate(X, y):
    """划分数据、训练模型并评估"""
    # 1. 划分训练集和测试集
    print("\n🔀 正在划分训练集和测试集 (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. 模型选择与训练 (使用随机森林)
    print("\n🌳 正在训练随机森林模型...")
    # 【S级任务】超参数调优
    # 定义要搜索的参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # 使用网格搜索进行交叉验证
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"✅ 训练完成！最佳参数: {grid_search.best_params_}")

    # 3. 在测试集上进行预测
    y_pred = best_model.predict(X_test)

    # 4. 性能评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 模型在测试集上的准确率 (Accuracy): {accuracy:.4f}")

    print("\n📋 详细分类报告:")
    print(classification_report(y_test, y_pred))

    return best_model, X_test, y_test, y_pred


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    """绘制并保存特征重要性图"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 从高到低排序

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 特征重要性图已保存至: {save_path}")


def main():
    # 1. 加载数据
    X, y = load_and_prepare_data(DATA_PATH)

    # 2. 训练与评估
    model, X_test, y_test, y_pred = train_and_evaluate(X, y)

    # 3. 可视化评估结果
    class_names = sorted(y.unique().astype(str).tolist())
    plot_confusion_matrix(y_test, y_pred, class_names, os.path.join(PLOT_SAVE_DIR, 'confusion_matrix.png'))

    # 4. 特征重要性分析
    plot_feature_importance(model, X.columns, os.path.join(PLOT_SAVE_DIR, 'feature_importance.png'))

    # 5. 保存模型
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n💾 模型已保存至: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()