# coordinator.py
import csv
import json
import os
import random
import torch
# coordinator.py
import time

from core.client import ByzantineParticipant
from filter.cluster import DBSCANClustering2D


class Coordinator:
    def __init__(self, participants, global_model, sp, tp, test_loader,
                 cluster_manager, agg_manager, reputation_manager,
                 num_rounds=10, k=0.8):
        self.participants = participants
        self.global_model = global_model
        self.sp = sp
        self.tp = tp
        self.test_loader = test_loader
        self.cluster_manager = cluster_manager
        self.agg_manager = agg_manager
        self.reputation_manager = reputation_manager
        self.num_rounds = num_rounds
        self.k = k  # 前 k 个高信誉度的客户端
        self.results = {
            "round": [],
            "round_time": [],  # 每轮耗时
            "accuracy": [],  # 每轮准确率
            "loss": [],  # 每轮损失
            "tpr": [],  # 每轮TPR
            "fpr": [],  # 每轮FPR
            "dbscan_error_rate": [],  # 每轮DBSCAN错误率
            "malicious_total": [],  # 每轮拜占庭个数
            "malicious_wrong": [],  # 每轮被选错的拜占庭
            "trusted_clients": [],  # 每轮可信客户端数量
            "overall_runtime": 0.0  # 总耗时（最终写入）
        }

    def run(self):
        total_start_time = time.time()  # 记录整体开始时间

        for r in range(self.num_rounds):
            round_start = time.time()
            print(f"\n=== Round {r + 1}/{self.num_rounds} ===")

            if self.cluster_manager != None:

                if r != 0:
                    selected = self._select_clients()
                else:
                    selected = [cid for cid in self.participants]
                print(f"[Round {r + 1}] Selected clients: {selected}")

                # === 2. 广播 & 本地训练
                self._broadcast(selected)
                self._local_train(selected)
                # === 3. DBSCAN 聚类
                distances = self.sp.compute_distances_pro(selected)
                angles = self.sp.compute_angles_pro(self.global_model.state_dict(), selected)
                distances_ = self.sp.adjusted_distances(selected)
                clustering = DBSCANClustering2D(distances, angles, eps=0.3, min_samples=2)
                clustering.fit()
                selected_cluster, mean_adjusted = self.sp.find_min_mean_adjusted_cluster(clustering.clustered_data,
                                                                                         distances_)
                print(f"✅ [Round {r + 1}] DBSCAN Trusted Cluster: {selected_cluster}")

                # === 4. 聚合可信客户端
                trusted_weights = [self.sp.received_shares[cid] for cid in selected_cluster]
                trusted_avg = self.agg_manager.aggregate(trusted_weights,selected_cluster)

                # === 5. 修正未选中客户端
                untrusted_clients = list(set(selected) - set(selected_cluster))
                self.agg_manager.adjust_untrusted(untrusted_clients, trusted_avg, self.participants, self.sp)

                # === 6. 再次聚合所有客户端
                all_clients = selected_cluster + untrusted_clients
                all_weights = [self.sp.received_shares[cid] for cid in all_clients]
                global_avg, selected_cluster = self.agg_manager.aggregate(all_weights, all_clients)
            else:
                # === 1. 候选客户端（信誉前k）
                selected = self.participants
                print(f"[Round {r + 1}] ")
                # === 2. 广播 & 本地训练
                self._broadcast(selected)
                self._local_train(selected)

                all_weights = [self.sp.received_shares[cid] for cid in selected]
                global_avg, selected_cluster = self.agg_manager.aggregate(all_weights, selected)

            # === 7. 更新全局模型
            self.global_model.load_state_dict(global_avg)
            self.sp._update_global_model(global_avg)

            if self.reputation_manager != None:
                # === 8. 更新信誉度
                self.reputation_manager.update_after_round(selected_cluster, untrusted_clients)

            # === 9. 评估性能
            acc, loss = self.evaluate_global(return_loss=True)

            # === 10. 统计恶意节点 & 计算指标
            malicious = [cid for cid in selected if isinstance(self.participants[cid], ByzantineParticipant)]
            benign = [cid for cid in selected if cid not in malicious]

            malicious_total = len(malicious)
            malicious_wrong = len(set(malicious) & set(selected_cluster))  # 被选为可信的恶意节点
            dbscan_error_rate = (malicious_wrong / malicious_total) if malicious_total > 0 else 0.0

            tpr, fpr = self.calculate_tpr_fpr(malicious, benign, selected_cluster)

            # === 11. 保存每轮结果
            round_time = time.time() - round_start
            self.results["round"].append(r + 1)
            self.results["round_time"].append(round_time)
            self.results["accuracy"].append(acc)
            self.results["loss"].append(loss)
            self.results["tpr"].append(tpr)
            self.results["fpr"].append(fpr)
            self.results["dbscan_error_rate"].append(dbscan_error_rate)
            self.results["malicious_total"].append(malicious_total)
            self.results["malicious_wrong"].append(malicious_wrong)
            self.results["trusted_clients"].append(len(selected_cluster))

            print(
                f"✅ [Round {r + 1}] acc={acc:.4f} loss={loss:.4f} TPR={tpr:.4f} FPR={fpr:.4f} DBSCAN错误率={dbscan_error_rate:.4f}")

        # === 12. 记录总耗时
        self.results["overall_runtime"] = time.time() - total_start_time
        print(f"🎯 全部 {self.num_rounds} 轮训练完成，总耗时 {self.results['overall_runtime']:.2f} 秒")

    def build_experiment_dir(self,config, base_dir="results/eval_logs"):
        parts = [
            config["dataset"],
            f"split_{config['split_mode']}",
            f"alpha_{config.get('dirichlet_alpha', 'NA')}",
            f"clients_{config['num_clients']}_rounds_{config['num_rounds']}",
            f"agg_{config['agg_method']}_cluster_{config['cluster_method']}",
            f"malicious_{config['malicious_ratio']}_{config['attack_type']}",
            f"lr_{config['learning_rate']}_ep_{config['local_epochs']}_bs_{config['batch_size']}"
        ]
        exp_dir = os.path.join(base_dir, *map(str, parts))
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def save_results_with_config(self, results, config, base_dir="results/eval_logs"):
        exp_dir = self.build_experiment_dir(config, base_dir)
        file_prefix = f"{config['agg_method']}_{config['cluster_method']}"

        json_path = os.path.join(exp_dir, f"{file_prefix}.json")
        csv_path = os.path.join(exp_dir, f"{file_prefix}.csv")

        # === 1️⃣ JSON 保存全部信息 ===
        import json
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ 实验结果 JSON 已保存: {json_path}")

        # === 2️⃣ 只取每轮 list 数据写 CSV ===
        per_round_keys = [k for k, v in results.items() if isinstance(v, list)]
        per_round_data = [results[k] for k in per_round_keys]

        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(per_round_keys)  # CSV 只写每轮数据
            for row in zip(*per_round_data):
                writer.writerow(row)

        print(f"✅ 实验结果 CSV 已保存: {csv_path}")
        print(f"✅ 总运行时间 overall_runtime = {results['overall_runtime']:.2f} 秒")

    # def save_results(self, save_path="results/exp1"):
    #     """保存实验结果"""
    #     import json, os
    #     os.makedirs(save_path, exist_ok=True)
    #
    #     # 保存为 JSON
    #     with open(os.path.join(save_path, "results.json"), "w") as f:
    #         json.dump(self.results, f, indent=2)
    #
    #     # 保存为 CSV
    #     import csv
    #     keys = self.results.keys()
    #     with open(os.path.join(save_path, "results.csv"), "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(keys)
    #         rows = zip(*[self.results[k] for k in keys])
    #         writer.writerows(rows)




    def evaluate_global(self, return_loss=False):
        """在 test_loader 上评估全局模型准确率与损失"""
        self.global_model.eval()
        correct, total = 0, 0
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(device)

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs, 1)
                loss = loss_fn(outputs, target)
                total_loss += loss.item()
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        if return_loss:
            return accuracy, total_loss / len(self.test_loader)  # 返回损失
        return accuracy


    def calculate_tpr_fpr(self, malicious, benign, filtered):
        """计算 TPR（恶意检测率）和 FPR（误杀率）"""
        true_pos = len(set(malicious) & set(filtered))  # 恶意且被过滤
        false_pos = len(set(benign) & set(filtered))  # 正常却被误杀

        tpr = true_pos / max(1, len(malicious))  # 恶意检测率
        fpr = false_pos / max(1, len(benign))  # 正常误杀率

        return tpr, fpr

    # def _select_clients(self):
    #     """随机选出一定比例的客户端"""
    #     m = max(1, int(len(self.participants) * self.k))
    #     return random.sample(list(self.participants.keys()), m)

    def _select_clients(self):
        """每轮根据信誉度排序，选前 k"""
        m = max(1, int(len(self.participants) * self.k))
        candidate_clients = self.reputation_manager.get_topk_clients(m)
        print(f"候选客户端(前{m}高信誉): {candidate_clients}")
        return candidate_clients

    def _broadcast(self, selected_clients):
        weights = self.global_model.state_dict()
        for cid in selected_clients:
            self.participants[cid].sync_global_model(weights)

    def _local_train(self, selected_clients):
        global_weights = self.global_model.state_dict()  # 传给拜占庭客户端
        for cid in selected_clients:
            client_obj = self.participants[cid]
            # 如果是拜占庭客户端，需要全局模型来生成攻击
            if isinstance(client_obj, ByzantineParticipant):
                w = client_obj.train_local(global_weights=global_weights)
            else:
                w = client_obj.train_local()
            self.sp.receive_shares(cid, w)



