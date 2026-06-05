# pipeline/clusterer.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

def select_k(embeddings: np.ndarray, n_papers: int) -> int:
    """
    Automatically select K using elbow method.
    - Minimum 2, maximum min(8, n_papers-1, 12)
    - For <6 papers, default K=2.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 4:
        return 2
    max_k = min(12, n_papers - 1, 8)
    if max_k < 2:
        max_k = 2

    if n_papers < 6:
        return 2

    inertias = []
    ks = range(2, max_k + 1)
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    # compute second derivative (approximate curvature)
    diff = np.diff(inertias)
    diff2 = np.diff(diff)
    if len(diff2) < 1:
        return 2
    elbow = np.argmax(diff2) + 2  # +2 because diff indexing
    return min(max(elbow, 2), max_k)

def cluster_embeddings(embeddings: np.ndarray, chunks: list, n_papers: int) -> list:
    """
    Stage 4: Run KMeans, select K, return cluster info.
    """
    if len(embeddings) == 0 or len(chunks) == 0:
        return []

    k = select_k(embeddings, n_papers)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    total_chunks = len(chunks)
    clusters = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        chunk_count = len(indices)
        percentage = (chunk_count / total_chunks) * 100
        sparse = percentage < 4.0

        # Find 5 chunks closest to centroid
        centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        dists = cosine_distances(embeddings[indices], centroid).flatten()
        closest_idx_in_cluster = np.argsort(dists)[:5]
        rep_indices = indices[closest_idx_in_cluster]
        representative_texts = [chunks[i]["text"] for i in rep_indices]

        clusters.append({
            "cluster_id": cluster_id,
            "chunk_count": chunk_count,
            "percentage": round(percentage, 2),
            "sparse": sparse,
            "representative_texts": representative_texts
        })

    return clusters
