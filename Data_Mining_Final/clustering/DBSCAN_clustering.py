import numpy as np
import pandas as pd
from collections import deque

# 讀取未知類別資料，包含 id 欄位
unknown_data_df = pd.read_csv("DNN_unknown.csv")
unknown_ids = unknown_data_df['id'].values
data = unknown_data_df.drop(columns=['id']).values

# 自訂參數
eps = 6.0     # 鄰域半徑
min_pts = 3   # 最小點數構成核心點

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def region_query(point_idx, data, eps):
    neighbors = []
    for i in range(len(data)):
        if i != point_idx and euclidean(data[point_idx], data[i]) < eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_idx, cluster_id, eps, min_pts):
    neighbors = region_query(point_idx, data, eps)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1  # mark as noise
        return False
    else:
        labels[point_idx] = cluster_id
        queue = deque(neighbors)
        
        while queue:
            current_idx = queue.popleft()
            
            # if current_idx is noise or unvisited
            if labels[current_idx] == -1:
                labels[current_idx] = cluster_id # assign noise point to current cluster
            elif labels[current_idx] == 0: # if unvisited, assign to cluster and explore its neighbors
                labels[current_idx] = cluster_id
                current_neighbors = region_query(current_idx, data, eps)
                if len(current_neighbors) >= min_pts: # if current_idx is a core point, add its neighbors to queue
                    queue.extend(current_neighbors)
        return True

def dbscan(data, eps, min_pts):
    labels = [0] * len(data)  # 0: unvisited, -1: noise, >0: cluster ID
    cluster_id = 0
    for i in range(len(data)):
        if labels[i] == 0:
            if expand_cluster(data, labels, i, cluster_id + 1, eps, min_pts):
                cluster_id += 1
    return labels

# 執行 DBSCAN
cluster_labels_numeric = dbscan(data, eps, min_pts)

# 將數值分群結果轉換為 'unknown X' 字串格式
# 噪音點 (-1) 也會被表示為 'unknown -1'
formatted_cluster_labels = ['unknown ' + str(label) for label in cluster_labels_numeric]

# 將分群結果與原始 ID 結合
clustering_result_df = pd.DataFrame({
    'id': unknown_ids,
    'Class': formatted_cluster_labels # Changed column name to 'Class'
})

# 儲存分群結果
clustering_result_df.to_csv("DBSCAN_clustering_result.csv", index=False)

print("clustering_result.csv has been saved with IDs and formatted cluster labels.")