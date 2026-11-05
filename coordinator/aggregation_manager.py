# aggregation_manager.py
import torch
import numpy as np

from adjustment.parameter_adjuster import ParameterAdjuster
EPS = 1e-12

class AggregationManager:
    def __init__(self, method="fedavg", beta=0.1, trim_ratio=0.1,
                 lr=0.1, num_iterations=50, distance_weight=0.1, angle_weight=2.0, angle_threshold=30,
                 fltrust_alpha=1.0):
        self.method = method
        self.beta = beta
        self.trim_ratio = trim_ratio
        self.fltrust_alpha = fltrust_alpha
        self.adjuster = ParameterAdjuster(
            lr=lr, num_iterations=num_iterations,
            distance_weight=distance_weight,
            angle_weight=angle_weight,
            angle_threshold=angle_threshold
        )
        # ---- FLTrust 运行上下文（默认空，外部配置一次）----
        self._fltrust_ctx = {
            "global_model": None,
            "root_loader": None,
            "loss_fn": None,
            "server_beta": 0.01,  # 服务器端本地学习率 β
            "Rl": 1,              # 服务器端本地步数
            "device": "cpu",
        }
        print(f"agg_method: {self.method}")

    # ---- 外部在训练前/每轮前调用一次，用于登记可信数据集等 ----
    def configure_fltrust(self, *, global_model, root_loader, loss_fn,
                          server_beta=0.01, Rl=1, device="cpu", alpha=None):
        self._fltrust_ctx.update({
            "global_model": global_model,
            "root_loader": root_loader,
            "loss_fn": loss_fn,
            "server_beta": server_beta,
            "Rl": Rl,
            "device": device
        })
        if alpha is not None:
            self.fltrust_alpha = alpha

    # ---- 扁平化/反扁平化（多方法可复用）----
    @staticmethod
    def _flatten_state_dict(sd):
        flats, shapes, keys = [], [], []
        for k, v in sd.items():
            t = v.detach().cpu().contiguous().view(-1)
            flats.append(t); shapes.append(v.shape); keys.append(k)
        return torch.cat(flats), shapes, keys

    @staticmethod
    def _unflatten_to_state_dict(vec, shapes, keys, template_sd):
        out, idx = {}, 0
        for k, shape in zip(keys, shapes):
            n = 1
            for s in shape: n *= s
            out[k] = vec[idx:idx+n].view(shape).type_as(template_sd[k])
            idx += n
        return out

    # ---- 仅供 FLTrust 调 g0：在 root dataset 上走 Rl 小步，返回 g0 向量 ----
    def _compute_server_update(self):
        ctx = self._fltrust_ctx
        model = ctx["global_model"]
        if model is None or ctx["root_loader"] is None or ctx["loss_fn"] is None:
            raise RuntimeError("FLTrust 未配置：请先调用 configure_fltrust(...)")

        device, beta, Rl = ctx["device"], ctx["server_beta"], ctx["Rl"]
        model = model.to(device); model.eval()

        # 备份当前全局 w
        w_global = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        w_vec0, shapes, keys = self._flatten_state_dict(w_global)

        # 只取一个 batch（配合 Rl=1 最简实现）
        for _ in range(Rl):
            for x, y in ctx["root_loader"]:
                x, y = x.to(device), y.to(device)
                model.zero_grad(set_to_none=True)
                logits = model(x)
                loss = ctx["loss_fn"](logits, y)
                loss.backward()
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p -= beta * p.grad
                break

        w_vecR, _, _ = self._flatten_state_dict(model.state_dict())
        g0_vec = w_vecR - w_vec0               # g0 = w_R - w
        return g0_vec, w_vec0, shapes, keys


    def aggregate(self, client_weights, client_ids):
        if not client_weights:
            return None, []

        client_weights = [{k: v.detach().cpu() for k, v in w.items()} for w in client_weights]

        if self.method == "fedavg":
            aggregated_model, selected_ids = self._fedavg(client_weights, client_ids)
        elif self.method == "median":
            aggregated_model, selected_ids = self._median(client_weights, client_ids)
        elif self.method == "trimmedmean":
            aggregated_model, selected_ids = self._trimmed_mean(client_weights, client_ids, self.trim_ratio)
        elif self.method == "krum":
            aggregated_model, selected_ids = self._krum(client_weights, client_ids)
        elif self.method == "bulyan":
            aggregated_model, selected_ids = self._bulyan(client_weights, client_ids)
        elif self.method == "fltrust":
            aggregated_model, selected_ids = self._fltrust(client_weights, client_ids)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        return aggregated_model, selected_ids

    # ----------------- 新增：FLTrust -----------------
    def _fltrust(self, client_weights, client_ids):
        # 1) 服务器在 root dataset 上得到 g0
        g0_vec, w_vec0, shapes, keys = self._compute_server_update()
        g0_norm = torch.norm(g0_vec) + EPS

        # 2) 计算每个客户端更新 gi = wi - w
        gi_list, norms_gi = [], []
        for wi in client_weights:
            wi_vec, _, _ = self._flatten_state_dict(wi)
            gi = wi_vec - w_vec0
            gi_list.append(gi)
            norms_gi.append(torch.norm(gi) + EPS)
        norms_gi = torch.stack([torch.tensor(x.item()) for x in norms_gi])

        # 3) 信任分 TSi = ReLU(cos(gi, g0))； 幅度归一化 \bar{g_i} = ||g0||/||gi|| * gi
        TS, gibar = [], []
        for gi, ngi in zip(gi_list, norms_gi):
            cos = torch.dot(gi, g0_vec) / (ngi * g0_norm)
            TS.append(torch.clamp(cos, min=0.0))                    # ReLU 截断
            gibar.append((g0_norm / ngi) * gi)                       # 归一化到与 g0 同范数
        TS = torch.stack(TS)

        # 4) 加权聚合 g = (Σ TS_i * \bar{g_i}) / Σ TS_i ；若全 0 则退化用 g0
        if TS.sum().item() <= 0:
            g_vec = g0_vec.clone()
            selected_ids = []
        else:
            g_sum = torch.zeros_like(w_vec0)
            for ts, gi_bar in zip(TS, gibar):
                g_sum += ts * gi_bar
            g_vec = g_sum / (TS.sum() + EPS)                        # 公式聚合
            selected_ids = [cid for cid, ts in zip(client_ids, TS) if ts.item() > 0]

        # 5) 全局更新：w ← w + α · g
        w_new_vec = w_vec0 + self.fltrust_alpha * g_vec
        template_sd = self._fltrust_ctx["global_model"].state_dict()
        w_new = self._unflatten_to_state_dict(w_new_vec, shapes, keys, template_sd)
        return w_new, selected_ids


    def adjust_untrusted(self, untrusted_clients, trusted_avg, participants, sp):
        adjusted_clients = []
        for cid in untrusted_clients:
            local_weights = participants[cid].model.state_dict()
            adjusted, final_angle = self.adjuster.adjust2target(local_weights, trusted_avg)
            sp.receive_shares(cid, adjusted)
            adjusted_clients.append(cid)
            print(f"[Adjuster] Adjusted {cid}, final angle={final_angle:.2f}")
        return adjusted_clients

    def _fedavg(self, client_weights, client_ids):
        # 这里选取所有的客户端进行聚合
        w_avg = {k: v.detach().cpu().clone() * 0.0 for k, v in client_weights[0].items()}
        for k in w_avg.keys():
            for w in client_weights:
                w_avg[k] += w[k].detach().cpu()
            w_avg[k] /= len(client_weights)
        return w_avg, client_ids  # 返回聚合模型和客户端ID

    def _median(self, client_weights, client_ids):
        keys = client_weights[0].keys()
        w_median = {}
        for k in keys:
            stacked = torch.stack([w[k].detach().cpu() for w in client_weights])
            w_median[k] = torch.median(stacked, dim=0).values
        return w_median, client_ids  # 返回聚合模型和客户端ID

    def _trimmed_mean(self, client_weights, client_ids, trim_ratio):
        n = len(client_weights)
        trim_k = int(n * trim_ratio)
        keys = client_weights[0].keys()
        w_trimmed = {}
        selected_ids = []

        for k in keys:
            stacked = torch.stack([w[k].detach().cpu() for w in client_weights])
            sorted_vals, indices = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k: n - trim_k]
            w_trimmed[k] = torch.mean(trimmed, dim=0)

            # 跟踪被保留（未被剔除）的客户端ID
            selected_ids = [client_ids[i] for i in indices[trim_k: n - trim_k]]

        return w_trimmed, selected_ids  # 返回聚合模型和参与聚合的客户端ID

    def _krum(self, client_weights, client_ids):
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
        return client_weights[best_idx], [client_ids[best_idx]]  # 返回聚合模型和选中的客户端ID

    def _bulyan(self, client_weights, client_ids):
        selected = []
        temp_weights = client_weights.copy()
        selected_ids = []
        for _ in range(len(client_weights) // 2):
            best = self._krum(temp_weights, client_ids)  # _krum 已经修复为 CPU
            selected.append(best[0])
            selected_ids.append(best[1][0])  # 获取客户端ID
            temp_weights.remove(best[0])
        return self._trimmed_mean(selected, selected_ids, trim_ratio=0.25)  # 返回聚合模型和客户端ID

