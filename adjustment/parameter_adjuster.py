# parameter_adjuster.py
import torch
import torch.nn.functional as F

class ParameterAdjuster:
    def __init__(self, lr=0.1, num_iterations=100,
                 distance_weight=0.1, angle_weight=2.0, angle_threshold=30):
        """
        lr: 调整学习率
        num_iterations: 调整迭代次数
        distance_weight: 距离损失权重
        angle_weight: 角度损失权重
        angle_threshold: 允许的最大角度差
        """
        self.lr = lr
        self.num_iterations = num_iterations
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.angle_threshold = angle_threshold

    def _flatten(self, weights):
        return torch.cat([v.flatten() for v in weights.values()])

    def _cosine_angle(self, v1, v2):
        dot = torch.dot(v1, v2)
        norm = torch.norm(v1) * torch.norm(v2) + 1e-8
        cos_theta = torch.clamp(dot / norm, -1.0, 1.0)
        return torch.rad2deg(torch.acos(cos_theta))

    def adjust2target(self, local_weights, target_weights):
        """
        local_weights: dict (未选中客户端的当前参数)
        target_weights: dict (可信客户端聚合后的参考参数)
        return: 调整后的权重 + 最终角度差
        """

        # 复制权重到 CPU (避免 device 冲突)
        adjusted = {k: v.detach().clone().cpu() for k, v in local_weights.items()}
        target = {k: v.detach().clone().cpu() for k, v in target_weights.items()}

        flat_target = self._flatten(target)

        # 调整迭代
        for _ in range(self.num_iterations):
            flat_local = self._flatten(adjusted)

            # === 距离损失（让本地参数接近目标参数）
            distance_loss = F.mse_loss(flat_local, flat_target)

            # === 角度损失（让角度差小于阈值）
            angle = self._cosine_angle(flat_local, flat_target)
            if angle > self.angle_threshold:
                angle_loss = (angle - self.angle_threshold) / self.angle_threshold
            else:
                angle_loss = torch.tensor(0.0)

            total_loss = self.distance_weight * distance_loss + self.angle_weight * angle_loss

            # === 梯度方向更新
            grad = torch.sign(flat_local - flat_target)  # 简化版梯度
            flat_local -= self.lr * grad * total_loss.item()

            # === 更新回 adjusted 权重
            idx = 0
            for k in adjusted.keys():
                numel = adjusted[k].numel()
                adjusted[k] = flat_local[idx:idx + numel].view_as(adjusted[k])
                idx += numel

        # 最终角度
        final_angle = self._cosine_angle(self._flatten(adjusted), flat_target)
        return adjusted, final_angle.item()
