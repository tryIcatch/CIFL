# main.py
from copy import deepcopy

from config_origin import config
from coordinator.aggregation_manager_ import AggregationManager
from coordinator.cluster_manager import ClusterManager
from coordinator.coordinator_origin import Coordinator
from coordinator.datamanager import DataManager
from coordinator.reputation_managerment import ReputationManager
from core.client import Participant, ByzantineParticipant
from core.server import ServiceProvider, ThirdParty
from model.models import MNISTModel, CIFAR10Model, SVHNCNN, CNN_32x32_BN


def get_model_by_dataset(dataset_name):
    """
    根据数据集名称选择模型
    """

    if dataset_name == "mnist" or dataset_name == "fashionmnist":
        return MNISTModel()
    elif dataset_name == "cifar10":
        return CIFAR10Model()
    elif dataset_name == "svhn":
        return SVHNCNN()
    elif dataset_name == 'gtsrb':
        return  CNN_32x32_BN()
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
            cluster_manager = ClusterManager()
            agg_manager = AggregationManager()
            reputation_manager = ReputationManager(participants)

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

