import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def run_experiment(train_path, test_path, experiment_name):
    print(f"\nExperiment {experiment_name}")

    # 讀取資料
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 切分 X 和 y
    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]
    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    # 訓練模型
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # 評估
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # confusion matrix 視覺化
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Experiment {experiment_name}")
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    plt.show()

# 執行實驗 A 和 B
run_experiment("C:/Users/ASUS/OneDrive/Desktop/Diabetes/expA/train_data_A.csv", "C:/Users/ASUS/OneDrive/Desktop/Diabetes/expA/test_data_A.csv", "A")
run_experiment("C:/Users/ASUS/OneDrive/Desktop/Diabetes/expB/train_data_B.csv", "C:/Users/ASUS/OneDrive/Desktop/Diabetes/expB/test_data_B.csv", "B")
