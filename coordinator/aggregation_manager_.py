# aggregation_manager.py
import torch
import numpy as np

from adjustment.parameter_adjuster import ParameterAdjuster


class AggregationManager:
    def __init__(self, method="fedavg", beta=0.1, trim_ratio=0.1,
                 lr=0.1, num_iterations=50, distance_weight=0.1, angle_weight=2.0, angle_threshold=30):
        self.method = method
        self.beta = beta
        self.trim_ratio = trim_ratio
        self.adjuster = ParameterAdjuster(
            lr=lr, num_iterations=num_iterations,
            distance_weight=distance_weight,
            angle_weight=angle_weight,
            angle_threshold=angle_threshold
        )

    def aggregate(self, client_weights):
        if not client_weights:
            return None

        # 聚合前先确保全部 CPU
        client_weights = [
            {k: v.detach().cpu() for k, v in w.items()}
            for w in client_weights
        ]

        if self.method == "fedavg":
            return self._fedavg(client_weights)
        elif self.method == "median":
            return self._median(client_weights)
        elif self.method == "trimmedmean":
            return self._trimmed_mean(client_weights, self.trim_ratio)
        elif self.method == "krum":
            return self._krum(client_weights)
        elif self.method == "bulyan":
            return self._bulyan(client_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

    def adjust_untrusted(self, untrusted_clients, trusted_avg, participants, sp):
        """
        修正未选中客户端的参数，并上传到服务器
        """
        adjusted_clients = []
        for cid in untrusted_clients:
            local_weights = participants[cid].model.state_dict()
            adjusted, final_angle = self.adjuster.adjust2target(local_weights, trusted_avg)
            sp.receive_shares(cid, adjusted)
            adjusted_clients.append(cid)
            print(f"[Adjuster] Adjusted {cid}, final angle={final_angle:.2f}")
        return adjusted_clients

    def _fedavg(self, client_weights):
        # 保证第一个客户端的权重作为模板，并且搬到 CPU
        w_avg = {k: v.detach().cpu().clone() * 0.0 for k, v in client_weights[0].items()}

        for k in w_avg.keys():
            for w in client_weights:
                # 聚合时强制 CPU，避免 cuda/cpu 混用
                w_avg[k] += w[k].detach().cpu()
            w_avg[k] /= len(client_weights)

        return w_avg

    def _median(self, client_weights):
        keys = client_weights[0].keys()
        w_median = {}
        for k in keys:
            stacked = torch.stack([
                w[k].detach().cpu() for w in client_weights
            ])
            w_median[k] = torch.median(stacked, dim=0).values
        return w_median

    def _trimmed_mean(self, client_weights, trim_ratio):
        n = len(client_weights)
        trim_k = int(n * trim_ratio)
        keys = client_weights[0].keys()
        w_trimmed = {}
        for k in keys:
            stacked = torch.stack([
                w[k].detach().cpu() for w in client_weights
            ])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k: n - trim_k]
            w_trimmed[k] = torch.mean(trimmed, dim=0)
        return w_trimmed

    def _krum(self, client_weights):
        flat_params = [
            torch.cat([p.detach().cpu().flatten() for p in w.values()])
            for w in client_weights
        ]
        n = len(flat_params)
        scores = []
        for i in range(n):
            dists = [
                torch.norm(flat_params[i] - flat_params[j]) ** 2
                for j in range(n) if i != j
            ]
            scores.append(sum(sorted(dists)[:n - 2]))
        best_idx = torch.argmin(torch.tensor(scores))
        return client_weights[best_idx]

    def _bulyan(self, client_weights):
        # 先选一半客户端（Multi-Krum）
        selected = []
        temp_weights = client_weights.copy()
        for _ in range(len(client_weights) // 2):
            best = self._krum(temp_weights)  # _krum 已经修复为 CPU
            selected.append(best)
            temp_weights.remove(best)
        # 然后在选出来的子集上再 TrimmedMean
        return self._trimmed_mean(selected, trim_ratio=0.25)



