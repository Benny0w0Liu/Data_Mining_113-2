import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# 讀取未知類別資料
unknown_data_df = pd.read_csv("DNN_unknown.csv")
unknown_ids = unknown_data_df['id'].values
data = unknown_data_df.drop(columns=['id']).values # K-Means algorithm needs only the features

data = data.astype(float)

# K-means
def initialize_centroids(data, k): # 隨機選擇 k 個點作為初始質心
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

def assign_to_clusters(data, centroids): # 將每個點分配給最近的質心，回傳一個陣列表示每個點所屬的 cluster ID
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, assignments, k): # 根據 cluster 分配，重新計算質心
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points_in_cluster = data[assignments == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster, axis=0)
        else:
            # 如果一個 cluster 變成了空的，則隨機選擇一個新的質心
            new_centroids[i] = data[np.random.choice(len(data))]
    return new_centroids

def calculate_wcss(data, centroids, assignments):
    # 計算 WCSS 用於評估 cluster 的緊密度
    wcss = 0
    for i, centroid in enumerate(centroids):
        points_in_cluster = data[assignments == i]
        wcss += np.sum(np.sum((points_in_cluster - centroid)**2, axis=1))
    return wcss

def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    # 1. 初始化質心
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iterations):
        # 2. 分配
        assignments = assign_to_clusters(data, centroids)
        
        # 3. 更新
        new_centroids = update_centroids(data, assignments, k)
        
        # 檢查質心是否收斂
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"K-Means converged after {iteration + 1} iterations.")
            break
        
        centroids = new_centroids
    else:
        print(f"K-Means reached max_iterations ({max_iterations}) without full convergence.")
        
    return assignments, centroids

# 決定最佳的 k 值 (Elbow Method)
wcss_values = []
max_k = 10 # 可根據資料量和預期 cluster 的數量調整這個值

print("Running K-Means for Elbow Method...")
for i in range(1, max_k + 1):
    # 由於 K-Means 對初始質心 sensitive，可以跑多次取平均 WCSS 或只取一次作為參考
    assignments, centroids = kmeans(data, i, max_iterations=100)
    wcss_values.append(calculate_wcss(data, centroids, assignments))

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss_values, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k (Manual K-Means)')
plt.grid(True)
plt.show()
# elbow 點對應的 k 值即為最佳數量

# 根據上一步 Elbow Method 的圖，手動設定最佳 k 值
optimal_k = 2
print(f"\nApply k={optimal_k} run K-Means clustering...")

cluster_labels_numeric, final_centroids = kmeans(data, optimal_k, max_iterations=200, tolerance=1e-5)

# 儲存分群結果
formatted_cluster_labels = ['unknown ' + str(label) for label in cluster_labels_numeric]

# 將分群結果與原始 ID 結合
clustering_result_df = pd.DataFrame({
    'id': unknown_ids,
    'Class': formatted_cluster_labels
})

# 儲存分群結果
clustering_result_df.to_csv("K-means_clustering_result.csv", index=False)

print("\nK-Means clustering ressult saved clustering_result_kmeans.csv")
print("\nclustering distribution: ")
unique_labels, counts = np.unique(cluster_labels_numeric, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  cluster {label} (unknown {label}): {count}")
