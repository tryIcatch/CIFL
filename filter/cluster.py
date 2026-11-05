# cluster.py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class DBSCANClustering2D:
    def __init__(self, distances, angles, eps=1, min_samples=2):
        """
        distances: dict {'P1': 16.5, 'P2': 12.3, ...}
        angles: dict {'P1': 2.8, 'P2': 3.1, ...}
        eps: DBSCAN epsilon
        min_samples: min samples for cluster
        """
        # 取交集，确保每个客户端都有距离和角度
        common_keys = set(distances.keys()) & set(angles.keys())
        self.keys = sorted(common_keys, key=lambda x: int(x[1:]))  # 排序P1,P2,P3...

        # 构造二维数组：[[dist, angle], ...]
        self.values = np.array([[distances[k], angles[k]] for k in self.keys])

        # 标准化避免尺度不一致
        self.values = StandardScaler().fit_transform(self.values)

        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.clustered_data = {}

    def fit(self):
        """执行DBSCAN聚类并过滤掉噪声点"""
        # 删除包含 NaN 的样本
        self.values = self.values[~np.isnan(self.values).any(axis=1)]

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = dbscan.fit_predict(self.values)

        # 初始化字典，存储有效簇的数据
        self.clustered_data = {}

        # 将每个数据点与其标签匹配，并过滤掉标签为 -1 的噪声点
        for key, label in zip(self.keys, self.labels):
            if label != -1:  # 只保存非噪声点
                if label not in self.clustered_data:
                    self.clustered_data[label] = []
                self.clustered_data[label].append(key)
        # print(self.clustered_data)
        return self

    def get_largest_cluster(self):
        """返回成员最多的簇"""
        max_label = None
        max_size = -1
        max_members = []

        for label, members in self.clustered_data.items():
            if len(members) > max_size:
                max_label = label
                max_size = len(members)
                max_members = members

        return {
            "label": max_label,
            "size": max_size,
            "data": max_members
        }



class HDBSCANClustering2D:
    def __init__(self, distances, angles, min_cluster_size=5):
        """
        2D HDBSCAN聚类
        """
        try:
            import hdbscan  # 需要额外安装 hdbscan
        except ImportError:
            raise ImportError("Please install hdbscan: pip install hdbscan")

        self.hdbscan_module = hdbscan
        common_keys = set(distances.keys()) & set(angles.keys())
        self.keys = sorted(common_keys, key=lambda x: int(x[1:]))

        self.values = np.array([[distances[k], angles[k]] for k in self.keys])
        self.values = StandardScaler().fit_transform(self.values)

        self.min_cluster_size = min_cluster_size
        self.labels = None
        self.clustered_data = {}

    def fit(self):
        clusterer = self.hdbscan_module.HDBSCAN(min_cluster_size=self.min_cluster_size)
        self.labels = clusterer.fit_predict(self.values)

        self.clustered_data = {}
        for key, label in zip(self.keys, self.labels):
            if label not in self.clustered_data:
                self.clustered_data[label] = []
            self.clustered_data[label].append(key)
        return self

    def get_largest_cluster(self):
        max_label = None
        max_size = -1
        max_members = []

        for label, members in self.clustered_data.items():
            if label == -1:
                continue
            if len(members) > max_size:
                max_label = label
                max_size = len(members)
                max_members = members

        return {
            "label": max_label,
            "size": max_size,
            "data": max_members
        }
