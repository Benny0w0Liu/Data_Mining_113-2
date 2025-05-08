import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load data
train_df = pd.read_csv(r"gene expression cancer RNA-Seq Data Set\train_data.csv")
train_labels_df = pd.read_csv(r"gene expression cancer RNA-Seq Data Set\train_label.csv")
test_df = pd.read_csv(r"gene expression cancer RNA-Seq Data Set\test_data.csv")

train_ids = train_df.iloc[:, 0]
test_ids = test_df.iloc[:, 0]
X_train = train_df.iloc[:, 1:]
X_test = test_df.iloc[:, 1:]
y_train = train_labels_df.iloc[:, 1]

# Set threshold for "unknown"
threshold = 0.6

# Train KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=21)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)
max_probs = np.max(probs, axis=1)
preds = model.predict(X_test)

print(max_probs)
final_preds = np.where(max_probs >= threshold, preds, "unknown")
result_df = pd.DataFrame({
    "id": test_ids,
    "prediction": final_preds
})
result_df.to_csv("classification/KNN_result.csv", index=False)
unknown_mask = final_preds == "unknown"
unknown_df = test_df[unknown_mask]
unknown_df.to_csv("classification/KNN_unknowns.csv", index=False)
print("shape of train data:\t",train_df.shape)
print("shape of KNN_unknowns:\t",unknown_df.shape)