# server.py
import torch
import math

class ServiceProvider:
    def __init__(self, participants):
        self.participants = participants  # 保存传入的参与者
        self.received_shares = {}  # {cid: weight_dict}

    def receive_shares(self, cid, weights):
        self.received_shares[cid] = weights

    def _flatten(self, weights):
        return torch.cat([w.detach().cpu().flatten() for w in weights.values()])

    def compute_distances_pro(self, selected_clients):
        flat = {cid: self._flatten(self.received_shares[cid]).detach().cpu()
                for cid in selected_clients}
        center = torch.stack(list(flat.values())).mean(dim=0)
        return {cid: torch.norm(flat[cid] - center, p=2).item() for cid in selected_clients}

    def compute_angles_pro(self, global_weights, selected_clients):
        """与全局模型向量的夹角"""
        # 统一放到 CPU
        global_flat = self._flatten(global_weights).detach().cpu()

        angles = {}
        for cid in selected_clients:
            local_flat = self._flatten(self.received_shares[cid]).detach().cpu()

            dot = torch.dot(local_flat, global_flat)
            norm = torch.norm(local_flat) * torch.norm(global_flat) + 1e-8
            cos_theta = torch.clamp(dot / norm, -1.0, 1.0)
            angle = torch.rad2deg(torch.acos(cos_theta))
            angles[cid] = angle.item()
        return angles

    def compute_average_model_pro(self, selected_clients):
        """FedAvg"""
        selected_weights = [self.received_shares[cid] for cid in selected_clients]
        w_avg = {k: torch.zeros_like(v) for k, v in selected_weights[0].items()}
        for k in w_avg.keys():
            for w in selected_weights:
                w_avg[k] += w[k]
            w_avg[k] /= len(selected_weights)
        return w_avg

    def _update_global_model(self, new_weights):
        """更新全局模型（Coordinator会调用）"""
        self.received_shares.clear()
        self.received_shares["global"] = new_weights

    def adjusted_distances(self, selected_clients):
        """计算每个客户端与全局模型的调整距离"""
        distances = self.compute_distances_pro(selected_clients)  # {pid: total_distance, ...}

        # 排序所有距离值，取中间那个（中位数）
        sorted_vals = sorted(distances.values())
        mid_idx = len(sorted_vals) // 2
        median_dist = sorted_vals[mid_idx]

        # 构造 pid -> (distance - median) 的映射
        adjusted = {pid: dist - median_dist for pid, dist in distances.items()}

        return adjusted

    def find_min_mean_adjusted_cluster(self, clustered_data, adjusted_distances):
        """
        参数:
          clustered_data: dict, e.g. {0: ['P0','P2',...], 1: [...], 3: [...]}
          adjusted_distances: dict, e.g. {'P0': 0.12, 'P2': -0.05, ...}
        返回:
          (min_cluster_id, member_list, mean_adjusted)
        """
        cluster_means = {}

        for cid, members in clustered_data.items():
            vals = [adjusted_distances[pid] for pid in members if pid in adjusted_distances]
            if not vals:
                continue
            cluster_means[cid] = sum(vals) / len(vals)

        if not cluster_means:
            return None, [], None

        # 找到平均 adjusted 最小的簇
        min_cid = min(cluster_means, key=lambda k: cluster_means[k])
        return clustered_data[min_cid], cluster_means[min_cid]

# server.py (继续)
class ThirdParty:
    def __init__(self):
        pass

    def verify(self, distances, angles):
        """未来可以扩展为独立验证节点"""
        return True
