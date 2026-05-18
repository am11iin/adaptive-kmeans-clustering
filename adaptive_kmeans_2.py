from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ===============================
# CONFIGURATION
# ===============================

DATA_PATH = "user_behavior_dataset.csv"
TARGET_COL = "User Behavior Class"
DROP_COLS = ["User ID"]
K_INIT = 3
RANDOM_STATE = 42
N_INIT = 10


# ===============================
# ALGORITHME
# ===============================

class AdaptiveDispersionKMeans:

    def __init__(
        self,
        k_init=3,
        max_iter=50,
        k_min=2,
        k_max=None,
        min_cluster_size=20,
        random_state=42,
        n_init=10,
    ):
        self.k_init = k_init
        self.max_iter = max_iter
        self.k_min = k_min
        self.k_max = k_max or max(2 * k_init, 10)
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.n_init = n_init

        self.cluster_centers_ = None
        self.labels_ = None
        self.n_clusters_ = None
        self.dispersions_ = None
        self.history_ = []

    def _reindex_labels(self, labels):
        unique_labels = np.unique(labels)
        mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        return np.array([mapping[label] for label in labels], dtype=int)

    # ===============================
    # DISPERSION
    # ===============================
    def _dispersion(self, X, labels, centers):
        dispersions = np.zeros(len(centers))

        for i, center in enumerate(centers):
            points = X[labels == i]
            if len(points) > 0:
                dispersions[i] = np.linalg.norm(points - center, axis=1).mean()

        return dispersions

    # ===============================
    # ASSIGNATION
    # ===============================
    def _assign(self, X, centers, dispersions=None):
        distances = pairwise_distances(X, centers)

        if dispersions is None or len(dispersions) != len(centers):
            return distances.argmin(axis=1)

        positive = dispersions[dispersions > 0]
        if len(positive) == 0:
            return distances.argmin(axis=1)

        weights = np.where(dispersions > 0, dispersions / positive.mean(), 1.0)
        return (distances * weights).argmin(axis=1)

    # ===============================
    # UPDATE CENTERS
    # ===============================
    def _update_centers(self, X, labels):
        return np.array([
            X[labels == i].mean(axis=0)
            for i in np.unique(labels)
        ])

    # ===============================
    # SPLIT
    # ===============================
    def _split(self, X, labels, centers):
        if len(centers) >= self.k_max:
            return centers

        dispersions = self._dispersion(X, labels, centers)
        positive = dispersions[dispersions > 0]

        if len(positive) == 0:
            return centers

        threshold = positive.mean()
        split_candidates = [
            i
            for i in range(len(centers))
            if len(X[labels == i]) >= self.min_cluster_size and dispersions[i] > threshold
        ]

        if not split_candidates:
            return centers

        # Split only the most dispersed cluster in one iteration to avoid
        # creating too many centers at the same time.
        split_idx = max(split_candidates, key=lambda i: dispersions[i])
        new_centers = []

        for i, center in enumerate(centers):
            points = X[labels == i]

            can_split = i == split_idx and len(new_centers) + 1 < self.k_max

            if can_split:
                sub_model = KMeans(
                    n_clusters=2,
                    random_state=self.random_state,
                    n_init=self.n_init,
                )
                sub_model.fit(points)
                new_centers.extend(sub_model.cluster_centers_)
            else:
                new_centers.append(center)

        return np.array(new_centers)

    # ===============================
    # MERGE
    # ===============================
    def _merge(self, centers):
        if len(centers) <= self.k_min:
            return centers

        distances = pairwise_distances(centers)
        np.fill_diagonal(distances, np.inf)

        nearest = np.min(distances, axis=1)
        valid_nearest = nearest[np.isfinite(nearest)]

        if len(valid_nearest) == 0:
            return centers

        # The median is more robust than the mean because it is less
        # sensitive to very distant cluster centers.
        threshold = np.median(valid_nearest)

        used = set()
        merged_centers = []

        for i in range(len(centers)):
            if i in used:
                continue

            group = [i]
            used.add(i)

            for j in range(i + 1, len(centers)):
                if j not in used and distances[i, j] < threshold:
                    group.append(j)
                    used.add(j)

            merged_centers.append(centers[group].mean(axis=0))

        merged_centers = np.array(merged_centers)

        return merged_centers if len(merged_centers) >= self.k_min else centers

    # ===============================
    # FIT
    # ===============================
    def fit(self, X):
        X = np.asarray(X, dtype=float)

        base_model = KMeans(
            n_clusters=self.k_init,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        labels = base_model.fit_predict(X)
        centers = base_model.cluster_centers_
        self.history_ = []

        for iteration in range(1, self.max_iter + 1):
            old_labels = labels.copy()
            old_k = len(centers)

            # Assign
            dispersions = self._dispersion(X, labels, centers)
            labels = self._assign(X, centers, dispersions)
            labels = self._reindex_labels(labels)
            centers = self._update_centers(X, labels)

            # Merge first so existing close centers are simplified before any
            # new split is created.
            centers = self._merge(centers)
            labels = self._assign(X, centers)
            labels = self._reindex_labels(labels)
            centers = self._update_centers(X, labels)

            # Split
            centers = self._split(X, labels, centers)
            labels = self._assign(X, centers)
            labels = self._reindex_labels(labels)
            centers = self._update_centers(X, labels)

            dispersions = self._dispersion(X, labels, centers)
            mean_dispersion = dispersions[dispersions > 0].mean()

            self.history_.append({
                "iteration": iteration,
                "clusters": len(centers),
                "dispersion_moyenne": float(mean_dispersion),
            })

            if old_k == len(centers) and np.array_equal(old_labels, labels):
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters_ = len(centers)
        self.dispersions_ = self._dispersion(X, labels, centers)

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Le modele doit etre entraine avant la prediction.")

        X = np.asarray(X, dtype=float)
        return self._assign(X, self.cluster_centers_, self.dispersions_)


# ===============================
# DATASET
# ===============================

def load_dataset(path, target_col=None, drop_cols=None):
    data = pd.read_csv(path)
    drop_cols = drop_cols or []

    y_true = None

    if target_col is not None and target_col in data.columns:
        y_true = data[target_col]
        drop_cols = drop_cols + [target_col]

    existing_drop_cols = [col for col in drop_cols if col in data.columns]
    X_raw = data.drop(columns=existing_drop_cols)

    X_raw = X_raw.fillna(X_raw.mode().iloc[0])
    X_encoded = pd.get_dummies(X_raw)
    X = StandardScaler().fit_transform(X_encoded)

    return X, y_true


# ===============================
# EVALUATION
# ===============================

def evaluate_clustering(X, labels, y_true=None):
    n_clusters = len(np.unique(labels))

    silhouette = silhouette_score(X, labels) if 1 < n_clusters < len(X) else np.nan

    results = {
        "clusters": n_clusters,
        "silhouette": silhouette,
    }

    if y_true is not None:
        results["ARI"] = adjusted_rand_score(y_true, labels)
        results["NMI"] = normalized_mutual_info_score(y_true, labels)

    return results


# ===============================
# VISUALISATION
# ===============================

def plot_comparison(X, labels_kmeans, centers_kmeans, labels_adaptive, centers_adaptive):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)

    X_2d = pca.fit_transform(X)
    centers_kmeans_2d = pca.transform(centers_kmeans)
    centers_adaptive_2d = pca.transform(centers_adaptive)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_kmeans, cmap="tab10")
    axes[0].scatter(centers_kmeans_2d[:, 0], centers_kmeans_2d[:, 1], c="black", marker="X")
    axes[0].set_title("K-means")

    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_adaptive, cmap="tab10")
    axes[1].scatter(centers_adaptive_2d[:, 0], centers_adaptive_2d[:, 1], c="black", marker="X")
    axes[1].set_title("Adaptive K-means (median merge)")

    plt.show()


# ===============================
# MAIN
# ===============================

def main():
    X, y_true = load_dataset(DATA_PATH, TARGET_COL, DROP_COLS)

    # Use the same fixed starting K for both methods so the adaptive version
    # can visibly show whether split/merge changes the clustering.
    kmeans = KMeans(n_clusters=K_INIT, random_state=RANDOM_STATE, n_init=N_INIT)
    labels_kmeans = kmeans.fit_predict(X)

    adaptive_model = AdaptiveDispersionKMeans(k_init=K_INIT)
    labels_adaptive = adaptive_model.fit_predict(X)

    plot_comparison(
        X,
        labels_kmeans,
        kmeans.cluster_centers_,
        labels_adaptive,
        adaptive_model.cluster_centers_,
    )


if __name__ == "__main__":
    main()
