
# 決策樹：糖尿病預測實驗
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 實驗 A：使用 train_data.csv + test_data.csv ----------
train_df_a = pd.read_csv("train_data.csv")
test_df_a = pd.read_csv("test_data.csv")

X_train_a = train_df_a.drop(columns=["Outcome"])
y_train_a = train_df_a["Outcome"]
X_test_a = test_df_a.drop(columns=["Outcome"])
y_test_a = test_df_a["Outcome"]

clf_a = DecisionTreeClassifier(max_depth=4, random_state=0)
clf_a.fit(X_train_a, y_train_a)
y_pred_a = clf_a.predict(X_test_a)
acc_a = accuracy_score(y_test_a, y_pred_a)
cm_a = confusion_matrix(y_test_a, y_pred_a)

print("===== 實驗 A =====")
print("準確率:", acc_a)
print("混淆矩陣:\n", cm_a)

# ---------- 實驗 B：使用 train_data (1).csv + test_data (1).csv ----------
train_df_b = pd.read_csv("train_data (1).csv")
test_df_b = pd.read_csv("test_data (1).csv")

X_train_b = train_df_b.drop(columns=["Outcome"])
y_train_b = train_df_b["Outcome"]
X_test_b = test_df_b.drop(columns=["Outcome"])
y_test_b = test_df_b["Outcome"]

clf_b = DecisionTreeClassifier(max_depth=4, random_state=0)
clf_b.fit(X_train_b, y_train_b)
y_pred_b = clf_b.predict(X_test_b)
acc_b = accuracy_score(y_test_b, y_pred_b)
cm_b = confusion_matrix(y_test_b, y_pred_b)

print("\n===== 實驗 B =====")
print("準確率:", acc_b)
print("混淆矩陣:\n", cm_b)

# ---------- 特徵重要性 ----------
def plot_feature_importance(model, features, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")

     # 設定橫軸範圍與刻度
    plt.xlim(0, 1)  # 橫軸從 0 到 1
    plt.xticks(np.arange(0, 1.1, 0.1))  # 每 0.1 一格

    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# ---------- 混淆矩陣 ----------
def plot_cm(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ---------- 圖輸出 ----------
plot_feature_importance(clf_a, X_train_a.columns, "Feature Importance - Experiment A")
plot_feature_importance(clf_b, X_train_b.columns, "Feature Importance - Experiment B")

plot_cm(cm_a, f"Experiment A Confusion Matrix (Acc: {acc_a:.2f})")
plot_cm(cm_b, f"Experiment B Confusion Matrix (Acc: {acc_b:.2f})")

