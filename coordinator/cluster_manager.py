# cluster_manager.py
from filter.cluster import DBSCANClustering2D, HDBSCANClustering2D


class ClusterManager:
    def __init__(self, method="dbscan", eps=0.1, min_samples=3, min_cluster_size=5):
        self.method = method
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size

    def filter_clients(self, distances, angles):
        if self.method == "dbscan":
            clusterer = DBSCANClustering2D(distances, angles, eps=self.eps, min_samples=self.min_samples)
        elif self.method == "hdbscan":
            clusterer = HDBSCANClustering2D(distances, angles, min_cluster_size=self.min_cluster_size)
        else:
            raise ValueError(f"Unknown clustering method {self.method}")

        clusterer.fit()
        largest_cluster = clusterer.get_largest_cluster()
        return largest_cluster['data']
