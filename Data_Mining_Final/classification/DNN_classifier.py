import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# 讀取資料，並移除 id 欄位
train_data = pd.read_csv("train_data.csv").drop(columns=['id']).values
train_label_raw = pd.read_csv("train_label.csv")["Class"].values

# 讀取 test data，保留 id 欄位以便後續輸出
test_data_df = pd.read_csv("test_data.csv")
test_ids = test_data_df['id'].values
test_data = test_data_df.drop(columns=['id']).values
test_label_raw = pd.read_csv("test_label.csv")["Class"].values

# 將訓練資料編碼
encoder = LabelEncoder()
encoder.fit(train_label_raw)
train_label = encoder.transform(train_label_raw)
num_classes = len(encoder.classes_)

# One-hot encode 訓練標籤
train_label_cat = tf.keras.utils.to_categorical(train_label, num_classes)

# 建立簡單 DNN 模型
model = models.Sequential([
    layers.Input(shape=(train_data.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(train_data, train_label_cat, epochs=20, batch_size=32, validation_split=0.2)

# 分離 test_data 中的已知與未知標籤
known_mask = np.isin(test_label_raw, encoder.classes_)
unknown_mask = ~known_mask

known_test_data = test_data[known_mask]
known_test_ids = test_ids[known_mask] # 提取已知樣本的 id
unknown_test_data = test_data[unknown_mask]
unknown_test_ids = test_ids[unknown_mask] # 提取未知樣本的 id

# 預測已知資料
pred_probs = model.predict(known_test_data)
pred_classes = np.argmax(pred_probs, axis=1)
pred_confidence = np.max(pred_probs, axis=1)

# 門檻設定：若最大信心度低於門檻值，視為未知類別
threshold = 0.8
pred_final_numeric = np.where(pred_confidence < threshold, -1, pred_classes)

# 將數值預測轉換為原始標籤，-1 轉換為 'unknown'
pred_final_labels = np.array([encoder.inverse_transform([p])[0] if p != -1 else 'unknown' for p in pred_final_numeric])

# 將最終結果儲存到 DataFrame
results_df = pd.DataFrame({'id': known_test_ids, 'label': pred_final_labels})

# 處理原始未知樣本
unknown_df_initial = pd.DataFrame({'id': unknown_test_ids, 'label': 'unknown'})

# 合併所有結果
final_output_df = pd.concat([results_df, unknown_df_initial]).sort_values(by='id').reset_index(drop=True)

# 儲存分類結果
final_output_df.to_csv("DNN_known.csv", index=False)

# 儲存未知樣本（供 DBSCAN 使用）
# 同時儲存數據和ID
unknown_samples_for_dbscan_df = test_data_df[test_data_df['id'].isin(final_output_df[final_output_df['label'] == 'unknown']['id'].values)].copy()
# Save the unknown data along with their IDs, for easy merging after clustering
unknown_samples_for_dbscan_df.to_csv("DNN_unknown.csv", index=False)

print("DNN_known.csv and DNN_unknown.csv have been saved in the desired format.")