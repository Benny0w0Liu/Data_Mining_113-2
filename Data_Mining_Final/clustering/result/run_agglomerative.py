import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from agglomerative import agglomerative

df = pd.read_csv("KNN_unknown.csv")
X_raw = df.select_dtypes(include=[np.number]).values

print("📌 原始資料維度：", X_raw.shape)

# ✅ PCA 降到 100 維以內（可調整）
pca = PCA(n_components=100)
X = pca.fit_transform(X_raw)

print("📌 降維後資料維度：", X.shape)

# ✅ 使用手刻分群
labels = agglomerative(X, k=2, linkage='single')

df['cluster'] = labels
df.to_csv("KNN_clustered.csv", index=False)

print("✅ 手刻 Agglomerative 分群完成，已存成 KNN_clustered.csv")
