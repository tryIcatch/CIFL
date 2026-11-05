# dataset_manager.py
import collections
import random
from collections import defaultdict
import os
import re
import kagglehub
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datasets import load_dataset
from ucimlrepo import fetch_ucirepo
from torchvision import transforms
# from transformers import BertTokenizer
from PIL import Image
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# class DatasetManager:
#     def __init__(self, dataset_name="mnist", num_participants=10, iid=True, batch_size=32):
#         self.dataset_name = dataset_name
#         self.num_participants = num_participants
#         self.iid = iid
#         self.batch_size = batch_size
#
#     def load(self):
#         transform = transforms.Compose([transforms.ToTensor()])
#         if self.dataset_name == "mnist":
#             train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
#             test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
#         else:
#             raise ValueError("Only MNIST implemented for now")
#
#         # IID 划分
#         indices = np.arange(len(train_dataset))
#         np.random.shuffle(indices)
#         split_indices = np.array_split(indices, self.num_participants)
#
#         participant_datasets = [Subset(train_dataset, idxs) for idxs in split_indices]
#         participant_loaders = [
#             DataLoader(ds, batch_size=self.batch_size, shuffle=True)
#             for ds in participant_datasets
#         ]
#         test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
#         return participant_loaders, test_loader

class TrafficSignDataset(Dataset):
    """
    csv_file: str 路径（包含 Path 与 ClassId 列）
    root_dir: 图像根目录
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)  # 这里读取
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 假设列名与常见 GTSRB CSV 一致：Path 与 ClassId
        rel_path = self.data_frame.iloc[idx, 7] if 'Path' not in self.data_frame.columns else self.data_frame.loc[idx, 'Path']
        label    = self.data_frame.iloc[idx, 6] if 'ClassId' not in self.data_frame.columns else self.data_frame.loc[idx, 'ClassId']

        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")  # 统一成RGB，避免通道不一致

        if self.transform:
            image = self.transform(image)
        return image, int(label)


class DataManager:
    def __init__(self, dataset_names, num_participants, split_method='iid', alpha=0.5,
                 batch_size=32, num_workers=0, seed=42, min_size_per_client=1):
        """
        dataset_names: List[str] 或 单个 str
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = dataset_names
        self.split_method = split_method
        self.num_participants = num_participants
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.min_size_per_client = min_size_per_client
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def load(self):
        train_datasets = {}
        test_loaders = {}

        for name in self.dataset_names:
            if name == "mnist":
                train_datasets[name] = self.load_mnist_dataset()
                test_loaders[name] = self.load_mnist_test()
            elif name == "fashionmnist":
                train_datasets[name] = self.load_fashionmnist_dataset()
                test_loaders[name] = self.load_fashionmnist_test()
            elif name == "cifar10":
                train_datasets[name] = self.load_cifar10_dataset()
                test_loaders[name] = self.load_cifar10_test()
            elif name == "svhn":
                train_datasets[name] = self.load_svhn_dataset()
                test_loaders[name] = self.load_svhn_test()
            elif name == 'gtsrb':
                train_datasets[name] = self.load_gtsrb_dataset()
                test_loaders[name] = self.load_gtsrb_test()
            else:
                raise ValueError(f"Unsupported dataset {name}")
        return train_datasets, test_loaders

    # ------------- datasets / tests -------------
    def load_gtsrb_dataset(self):
        gtsrb_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        dataset = TrafficSignDataset(
            csv_file='./data/gtsrb/Train.csv',  # 注意这里传路径字符串
            root_dir='./data/gtsrb',
            transform=gtsrb_transform
        )
        return dataset  # 和 MNIST 保持接口一致：返回训练 Dataset

    def load_gtsrb_test(self):
        gtsrb_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        dataset = TrafficSignDataset(
            csv_file='./data/gtsrb/Test.csv',  # 这里也是路径字符串
            root_dir='./data/gtsrb',
            transform=gtsrb_transform
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def load_mnist_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        return datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    def load_mnist_test(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def load_fashionmnist_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        return datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    def load_fashionmnist_test(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def load_cifar10_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])
        return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    def load_cifar10_test(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def load_svhn_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4377, 0.4438, 0.4728),
                                                             (0.1980, 0.2010, 0.1970))])
        return datasets.SVHN(root='./data', split='train', download=True, transform=transform)

    def load_svhn_test(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4377, 0.4438, 0.4728),
                                                             (0.1980, 0.2010, 0.1970))])
        test_data = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    # ------------- split -------------
    def split(self, train_datasets: dict):
        """返回 {dataset_name: [DataLoader for client 0..N-1]}，各客户端样本互斥"""
        participant_loaders = {}

        for name, ds in train_datasets.items():
            if hasattr(ds, 'targets'):
                targets = np.array(ds.targets)
            elif hasattr(ds, 'labels'):
                targets = np.array(ds.labels)
            else:
                # 慢但通用的回退
                targets = np.array([ds[i][1] for i in range(len(ds))])

            if self.split_method == 'iid':
                splits = self._iid_split(ds)
            elif self.split_method == 'non_iid':
                splits = self._dirichlet_split(ds, targets)
            elif self.split_method == 'twoclass':
                splits = self._twoclass_disjoint_split(ds, targets)
            else:
                raise ValueError("Invalid split method. Choose 'iid' | 'non_iid' | 'twoclass'.")

            loaders = [
                DataLoader(subset, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers)
                for subset in splits
            ]
            participant_loaders[name] = loaders

        return participant_loaders

    def _iid_split(self, dataset):
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        split_idx = np.array_split(idx, self.num_participants)
        return [Subset(dataset, s.tolist()) for s in split_idx]

    def _dirichlet_split(self, dataset, targets):
        num_classes = int(np.max(targets)) + 1
        class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
        client_indices = [[] for _ in range(self.num_participants)]

        # 反复采样直到每个客户端 >= min_size_per_client
        while True:
            tmp = [[] for _ in range(self.num_participants)]
            for c, idxs in enumerate(class_indices):
                idxs = np.array(idxs)
                np.random.shuffle(idxs)
                props = np.random.dirichlet([self.alpha] * self.num_participants)
                # 按比例切分
                cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
                chunks = np.split(idxs, cuts)
                for cid, ch in enumerate(chunks):
                    tmp[cid].extend(ch.tolist())
            sizes = [len(t) for t in tmp]
            if min(sizes) >= self.min_size_per_client:
                client_indices = tmp
                break

        return [Subset(dataset, idxs) for idxs in client_indices]

    def _twoclass_disjoint_split(self, dataset, targets):
        """
        修复后的 twoclass：先把每个类别的索引随机平均切给各客户端（互斥），
        然后每个客户端随机选 2 个“保留类别”，仅保留其在这两个类别中被分到的那部分样本。
        这样既 two-class，又保证不重叠。
        """
        num_classes = int(np.max(targets)) + 1
        # 1) 类别内先均匀分片到各客户端（互斥）
        per_client_indices = [[] for _ in range(self.num_participants)]
        for c in range(num_classes):
            idxs = np.where(targets == c)[0]
            np.random.shuffle(idxs)
            chunks = np.array_split(idxs, self.num_participants)
            for cid, ch in enumerate(chunks):
                per_client_indices[cid].extend(ch.tolist())

        # 2) 每个客户端选 2 类，只保留这 2 类内它本来就分到的那部分
        final_indices = []
        for cid in range(self.num_participants):
            chosen = np.random.choice(np.arange(num_classes), size=min(2, num_classes), replace=False)
            chosen = set(chosen.tolist())
            keep = [i for i in per_client_indices[cid] if targets[i] in chosen]
            # 保底：若极端情况下为空，从自己已有里随机采样若干
            if len(keep) == 0 and len(per_client_indices[cid]) > 0:
                k = min( max(self.min_size_per_client, 32), len(per_client_indices[cid]) )
                keep = random.sample(per_client_indices[cid], k)
            final_indices.append(keep)

        return [Subset(dataset, idxs) for idxs in final_indices]

        # -------------------- Root dataset builder --------------------

    def build_root_loader(self, name, root_size=100, batch_size=None,
                          per_class=False, seed=None):
        """
        从训练集独立抽样 root_size 个干净样本，构造可信 root dataloader。
        - per_class=True：尽量按“各类均衡”抽样（适用于分类任务）
        """
        if batch_size is None:
            batch_size = self.batch_size
        if seed is None:
            seed = self.seed
        g = torch.Generator().manual_seed(seed)

        if name not in self.dataset_names:
            raise ValueError(f"Root dataset name {name} not loaded")

        train_ds = None
        # 你的 load() 返回的是 “训练 Dataset 字典”
        # 请先在外面调用 load() 再调用本函数
        # 这里不直接调用 load()，避免重复下载/构造

        # 为简单，这里从 self.load() 调用者处传入的缓存取：
        # 你在 main 中已有 train_datasets = dm.load()[0]
        # 因为这里拿不到那个局部变量，所以提供一个 setter（见下）
        if not hasattr(self, "_train_cache") or name not in self._train_cache:
            raise RuntimeError("Please call dm.set_train_cache(train_datasets) before build_root_loader")

        train_ds = self._train_cache[name]

        # 取标签向量（兼容 MNIST/Fashion/CIFAR/SVHN/GTSRB 自定义数据集）
        labels = None
        for attr in ("targets", "labels", "ys", "y"):
            if hasattr(train_ds, attr):
                labels = getattr(train_ds, attr)
                if isinstance(labels, list):
                    labels = torch.tensor(labels)
                break
        if labels is None:
            # 如果自定义 Dataset 没有这些属性，请在自定义类里暴露 .labels
            raise RuntimeError(f"Dataset {name} has no labels attribute for per-class sampling")

        N = len(train_ds)
        root_size = min(root_size, N)

        if not per_class:
            # 简单均匀随机
            idx = torch.randperm(N, generator=g)[:root_size].tolist()
        else:
            # 尽量各类均衡抽样
            idx = []
            labels = labels.view(-1).cpu()
            cls_ids = collections.defaultdict(list)
            for i, y in enumerate(labels.tolist()):
                cls_ids[int(y)].append(i)
            n_cls = len(cls_ids)
            per = max(1, root_size // n_cls)
            for y, arr in cls_ids.items():
                arr = torch.tensor(arr)
                take = min(per, len(arr))
                choose = arr[torch.randperm(len(arr), generator=g)[:take]].tolist()
                idx.extend(choose)
            # 不足则补齐
            if len(idx) < root_size:
                rest = root_size - len(idx)
                pool = torch.randperm(N, generator=g).tolist()
                for i in pool:
                    if i not in idx:
                        idx.append(i)
                        if len(idx) >= root_size:
                            break

        subset = Subset(train_ds, idx)
        return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 可注入训练集缓存，便于 build_root_loader 使用

    def set_train_cache(self, train_datasets: dict):
        self._train_cache = train_datasets
