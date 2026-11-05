# reputation_manager.py
class ReputationManager:
    def __init__(self, participants, init_score=1.0, blacklist_threshold=0.2, delta_pos=0.05, delta_neg=0.1):
        """
        初始化信誉度管理器
        init_score: 初始信誉度
        blacklist_threshold: 进入黑名单的信誉度阈值
        delta_pos: 每轮被选中客户端增加的信誉度
        delta_neg: 被 DBSCAN 过滤的客户端扣除的信誉度
        """
        self.reputation = {cid: init_score for cid in participants.keys()}
        self.blacklist_threshold = blacklist_threshold
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.participants = participants

    def get_topk_clients(self, k):
        """根据信誉度选择前 k 个客户端"""
        sorted_clients = sorted(self.reputation.items(), key=lambda x: x[1], reverse=True)
        topk_clients = [cid for cid, score in sorted_clients[:k] if score >= self.blacklist_threshold]
        return topk_clients

    def update_after_round(self, trusted_clients, filtered_clients):
        """更新客户端信誉度：选中的加分，过滤掉的扣分"""
        for cid in trusted_clients:
            self.reputation[cid] += self.delta_pos
        for cid in filtered_clients:
            self.reputation[cid] -= self.delta_neg

        # 如果信誉度低于阈值，进入黑名单
        for cid, score in self.reputation.items():
            if score < self.blacklist_threshold:
                self.reputation[cid] = 0.0  # 永久拉黑，不能再参与训练
