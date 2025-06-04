import numpy as np

def _pairwise_dist(X):
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    return D

def agglomerative(X, k=5, linkage='average'):
    n = len(X)
    clusters = {i: [i] for i in range(n)}
    D = _pairwise_dist(X)

    valid = np.ones(n, dtype=bool)  # 標記哪些 cluster 還活著

    while sum(valid) > k:
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if i == j or not valid[i] or not valid[j]:
            D[i, j] = np.inf
            continue

        clusters[i] += clusters[j]
        valid[j] = False

        for m in range(n):
            if not valid[m] or m == i:
                continue
            a = np.array(clusters[i])
            b = np.array(clusters[m])
            if linkage == 'single':
                dist = np.min(D[np.ix_(a, b)])
            elif linkage == 'complete':
                dist = np.max(D[np.ix_(a, b)])
            elif linkage == 'average':
                dist = np.mean(D[np.ix_(a, b)])
            else:
                raise ValueError("linkage 必須是 single, complete, average")
            D[i, m] = D[m, i] = dist

        D[j, :] = D[:, j] = np.inf

    # 把 cluster 結果整理成 labels
    labels = np.empty(n, dtype=int)
    cluster_ids = [i for i, v in enumerate(valid) if v]
    for cid, cluster_id in enumerate(cluster_ids):
        for idx in clusters[cluster_id]:
            labels[idx] = cid
    return labels
