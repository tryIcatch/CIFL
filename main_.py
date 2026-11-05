# main.py
from copy import deepcopy
import torch
from config_ import config
from coordinator.aggregation_manager import AggregationManager
from coordinator.cluster_manager import ClusterManager
from coordinator.coordinator import Coordinator
from coordinator.datamanager import DataManager
from coordinator.reputation_managerment import ReputationManager
from core.client import Participant, ByzantineParticipant
from core.server import ServiceProvider, ThirdParty
from model.models import MNISTModel, CIFAR10Model, SVHNCNN


def get_model_by_dataset(dataset_name):
    if dataset_name in ("mnist", "fashionmnist"):
        return MNISTModel()
    elif dataset_name == "cifar10":
        return CIFAR10Model()
    elif dataset_name == "svhn":
        return SVHNCNN()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


for dataset in config["dataset"]:
    for split_mode in config["split_mode"]:
        for malicious_ratio in config["malicious_ratio"]:
            print(f"dataset: {dataset} split_mode: {split_mode} malicious_ratio: {malicious_ratio}")
            # 选择当前配置的副本
            current_config = deepcopy(config)
            current_config["dataset"] = [dataset]  # 这里是单个数据集
            current_config["split_mode"] = split_mode
            current_config["malicious_ratio"] = malicious_ratio
            dm = DataManager(
                dataset_names=current_config["dataset"],  # 可为 str 或 List[str]
                num_participants=current_config['num_clients'],
                split_method=split_mode,
                alpha=current_config.get('dirichlet_alpha', 0.5),
                batch_size=current_config.get('batch_size', 32),
                num_workers=current_config.get('num_workers', 0),
                seed=current_config.get('seed', 42),
                min_size_per_client=current_config.get('min_size_per_client', 1)
            )

            train_datasets, test_loaders = dm.load()
            dm.set_train_cache(train_datasets)

            participant_loaders = dm.split(train_datasets)
            # 初始化参与者
            participants = {}
            for i in range(current_config["num_clients"]):
                model = get_model_by_dataset(dataset)  # 动态选择模型
                loader = participant_loaders[dataset][i]

                if i < int(current_config["num_clients"] * current_config["malicious_ratio"]):
                    participants[f"P{i}"] = ByzantineParticipant(
                        train_loader=loader,
                        model=model,
                        attack_type=current_config["attack_type"]
                    )
                else:
                    participants[f"P{i}"] = Participant(
                        train_loader=loader,
                        model=model
                    )

            # 继续进行实验并保存结果
            global_model = model
            sp = ServiceProvider(participants)
            tp = ThirdParty()

            if current_config['cluster_method'] != 'none':
                cluster_manager = ClusterManager()
                reputation_manager = ReputationManager(participants)
            else:
                cluster_manager = None
                reputation_manager =None
            agg_manager = AggregationManager(method=current_config.get("agg_method","fedavg"), fltrust_alpha=1.0)

            # 构造可信 root loader：100 个样本，按类别均衡（可改）
            root_loader = dm.build_root_loader(
                name=dataset,
                root_size=current_config.get("root_size", 100),
                batch_size=current_config.get("root_batch_size", 32),
                per_class=current_config.get("root_per_class", True)
            )

            loss_fn = torch.nn.CrossEntropyLoss()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 只需配置一次，aggregate 签名不变
            agg_manager.configure_fltrust(
                global_model=global_model,
                root_loader=root_loader,
                loss_fn=loss_fn,
                server_beta=current_config.get("server_beta", 0.01),
                Rl=current_config.get("server_Rl", 1),
                device=device,
                alpha=current_config.get("fltrust_alpha", 1.0)
            )

            coord = Coordinator(
                participants=participants,
                global_model=global_model,
                sp=sp,
                tp=tp,
                test_loader=test_loaders[dataset],
                cluster_manager=cluster_manager,
                agg_manager=agg_manager,
                reputation_manager=reputation_manager,
                num_rounds=current_config["num_rounds"],
                k=0.8
            )

            coord.run()
            coord.save_results_with_config(coord.results, current_config)


# # 加载数据集
# dm = DatasetManager(
#     dataset_name=config["dataset"],
#     num_participants=config["num_clients"],
#     iid=(config["split_mode"] == "iid"),
#     alpha=config["dirichlet_alpha"],
#     batch_size=config["batch_size"]
# )
# participant_loaders, test_loader = dm.load()
#
# attack_type = config["attack_type"]
# participants = {}
#
# for i in range(config["num_clients"]):
#     loader = participant_loaders[i]
#     model = MNISTModel()
#
#     if i < int(config["num_clients"] * config["malicious_ratio"]):
#         participants[f"P{i}"] = ByzantineParticipant(
#             train_loader=loader,
#             model=model,
#             attack_type=attack_type
#         )
#     else:
#         participants[f"P{i}"] = Participant(
#             train_loader=loader,
#             model=model
#         )
#
# reputation_manager = ReputationManager(participants)
#
# # 创建服务器、聚合管理器、聚类管理器
# sp = ServiceProvider()
# tp = ThirdParty()
# agg_manager = AggregationManager(method=config["agg_method"])
# cluster_manager = ClusterManager(method=config["cluster_method"])
#
# # 创建协调器
# coord = Coordinator(
#     participants, MNISTModel(), sp, tp, test_loader,
#     cluster_manager, agg_manager, reputation_manager,
#     num_rounds=config["num_rounds"], k=0.8
# )
#
# coord.run()
# coord.save_results_with_config(coord.results, config)

