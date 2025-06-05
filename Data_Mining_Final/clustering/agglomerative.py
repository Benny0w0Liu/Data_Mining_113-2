import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


os.makedirs("result", exist_ok=True)
merge_log = []
merge_step = 0

df = pd.read_csv("KNN_unknown.csv")
X_ids = df.iloc[:, 0]
X = df.iloc[:, 1:].to_numpy()
n = X.shape[0]

z_scores = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
outliers = np.abs(z_scores) > 3
median = np.median(X, axis=0)
X[outliers] = median[np.where(outliers)[1]]

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std

num_clusters = 2
cluster_labels = np.arange(n)
clusters = {i: [i] for i in range(n)}
centroids = {i: X[i].copy() for i in range(n)}
sizes = {i: 1 for i in range(n)}
active_clusters = set(range(n))


next_cluster_id = n
linkage = []  
while len(active_clusters) > num_clusters:
    min_increase = np.inf
    best_pair = None
    active_list = list(active_clusters)

    for i in range(len(active_list)):
        for j in range(i + 1, len(active_list)):
            ci = active_list[i]
            cj = active_list[j]

            ni, nj = sizes[ci], sizes[cj]
            mi, mj = centroids[ci], centroids[cj]

            dist = np.sum((mi - mj) ** 2)
            increase = (ni * nj) / (ni + nj) * dist

            if increase < min_increase:
                min_increase = increase
                best_pair = (ci, cj)

    ci, cj = best_pair

    merge_log.append({
        "id": next_cluster_id,
        "left": ci,
        "right": cj,
        "distance": min_increase
    })
    linkage.append((ci, cj, min_increase, sizes[ci] + sizes[cj]))

    new_points = clusters[ci] + clusters[cj]
    clusters[next_cluster_id] = new_points
    centroids[next_cluster_id] = np.mean(X[new_points], axis=0)
    sizes[next_cluster_id] = len(new_points)

    for idx in new_points:
        cluster_labels[idx] = next_cluster_id

    active_clusters.remove(ci)
    active_clusters.remove(cj)
    active_clusters.add(next_cluster_id)

    clusters.pop(ci)
    clusters.pop(cj)
    centroids.pop(ci)
    centroids.pop(cj)
    sizes.pop(ci)
    sizes.pop(cj)

    next_cluster_id += 1

unique_labels = np.unique(cluster_labels)
label_mapping = {label: i for i, label in enumerate(unique_labels)}
final_labels = np.array([label_mapping[label] for label in cluster_labels])


result_df = pd.DataFrame({"ID": X_ids, "Cluster": final_labels})
result_df = result_df.sort_values(by="ID", key=lambda x: x.str.extract(r'(\d+)').astype(int).squeeze())
result_df.to_csv("result/KNN_agglomerative.csv", index=False)


children = {}
for m in merge_log:
    children[m["id"]] = (m["left"], m["right"])


def get_leaf_order(node):
    if node < n:
        return [node]
    left, right = children[node]
    return get_leaf_order(left) + get_leaf_order(right)

leaf_order = get_leaf_order(next_cluster_id - 1)
leaf_pos = {leaf: i * 10 for i, leaf in enumerate(leaf_order)}


fig, ax = plt.subplots(figsize=(16, 6))
positions = {}

for log in merge_log:
    c1, c2 = log["left"], log["right"]
    new_id = log["id"]
    y = log["distance"]

    x1 = positions[c1][0] if c1 in positions else leaf_pos.get(c1, None)
    x2 = positions[c2][0] if c2 in positions else leaf_pos.get(c2, None)
    y1 = positions[c1][1] if c1 in positions else 0
    y2 = positions[c2][1] if c2 in positions else 0

    if x1 is None or x2 is None:
        continue

    xc = (x1 + x2) / 2
    positions[new_id] = (xc, y)

    ax.plot([x1, x1, x2, x2], [y1, y, y, y2], c="black", linewidth=1)

ax.set_title("KNN-Agglomerative")
ax.set_xlabel("Samples")
ax.set_ylabel("Distance")
ax.get_xaxis().set_visible(False)
ax.grid(False)

plt.tight_layout()
plt.savefig("result/agglomerative_KNN.png")
plt.show()
