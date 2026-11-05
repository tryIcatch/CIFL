# client.py
import torch
import torch.nn.functional as F

class Participant:
    def __init__(self, train_loader, model, local_epochs=1, lr=0.01):
        self.train_loader = train_loader
        self.model = model
        self.local_epochs = local_epochs
        self.lr = lr

    def sync_global_model(self, global_weights):
        self.model.load_state_dict(global_weights)

    def train_local(self):
        """正常训练"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        return self.model.state_dict()

class ByzantineParticipant(Participant):
    def __init__(self, train_loader, model, attack_type="lie", device="cpu", **kwargs):
        super().__init__(train_loader, model, **kwargs)
        self.attack_type = attack_type
        self.device = device

    def train_local(self, global_weights=None, noise_std=0.01):
        """
        拜占庭客户端本地上传权重
        - global_weights: 全局模型参数 (dict)
        - noise_std: IPM 高斯噪声标准差
        """
        # 正常训练
        w = super().train_local()

        # 如果没指定攻击，就上传正常训练结果
        if self.attack_type is None:
            return w

        # 执行不同拜占庭攻击
        if self.attack_type == "lie":
            return self._lie_attack(w, global_weights)

        elif self.attack_type == "ipm":
            return self._ipm_attack(w, global_weights, noise_std)

        elif self.attack_type == "sign-flip":
            return {k: -1.0 * v for k, v in w.items()}

        elif self.attack_type == "gaussian":
            return {k: v + 0.5 * torch.randn_like(v) for k, v in w.items()}

        else:
            print(f"⚠️ Unknown attack type {self.attack_type}, default benign")
            return w

    def _lie_attack(self, w, global_weights, z=5.0):
        """
        Lie Attack: 伪造接近全局模型 + 固定偏移的参数，扰乱聚合方向
        mu = 全局模型参数
        noise = z * sigma( local - mu )
        """
        lie_w = {}
        for key in w:
            # 跳过非浮点参数（BN 计数器等）
            if not torch.is_floating_point(w[key]):
                lie_w[key] = w[key]
                continue

            local = w[key].detach().cpu().float()
            mu = global_weights[key].detach().cpu().float()  # 全局参考
            sigma = torch.std(local - mu)  # 统计差异幅度

            noise = z * sigma  # 固定倍数的偏移
            lie_param = mu + noise  # 伪造看似合理但带偏移的更新
            lie_w[key] = lie_param.to(w[key].device).type_as(w[key])  # 保留原始数据类型

        print("⚠️ [LieAttack] 上传撒谎型伪造更新(接近全局+扰动)")
        return lie_w

    def _ipm_attack(self, w, global_weights, noise_std=0.01):
        """
        IPM Attack: 根据全局模型反方向上传 + 高斯噪声干扰
        让聚合方向完全逆向
        """
        ipm_w = {}
        for key in w:
            if not torch.is_floating_point(w[key]):
                ipm_w[key] = w[key]
                continue

            benign = global_weights[key].detach().to(w[key].device)
            noise = torch.normal(0, noise_std, size=w[key].shape).to(w[key].device)
            ipm_w[key] = -1.0 * benign + noise  # 反向上传
        print("⚠️ [IPM Attack] 上传逆向扰动更新")
        return ipm_w
