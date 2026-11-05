# config.py

config = {
    # 数据集设置
    "dataset": [ "mnist","fashionmnist"],
    "split_mode": ["iid","non_iid"],     # "iid", "dirichlet", "twoclass"（每个客户端2个类的数据）
    "dirichlet_alpha": 0.1,  # Dirichlet 分布的 alpha 参数，越小，数据越不均匀

    # 参与者和训练设置
    "num_clients":30,  # 参与者数量
    "num_rounds": 30,  # 训练轮次
    "batch_size": 32,  # 每个客户端的批次大小

    # 聚合方法
    "agg_method": "fltrust",  # "fedavg", "median", "trimmedmean", "krum", "bulyan"
    "cluster_method": "none",  # "dbscan", "hdbscan"

    # 信誉度管理设置
    "reputation": "none",

    # 训练超参数
    "learning_rate": 0.01,  # 本地训练的学习率
    "local_epochs": 5,  # 每个客户端本地训练轮数

    "malicious_ratio": [0.1,0.2,0.3,0.4],  # 恶意节点比例
    "attack_type": "sign-flip",  # "sign-flip" 或 "lie"
}

