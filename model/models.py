# model.py
import torch.nn as nn
import torch.nn.functional as F
import  torch

class SVHNCNN(nn.Module):
    def __init__(self):
        super(SVHNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)  # SVHN 是数字分类，0~9

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool2(F.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class GTSRBCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(GTSRBCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 输入3通道 (RGB/灰度转3通道)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16 -> 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_32x32_BN(nn.Module):
    def __init__(self, num_classes=43):
        super(CNN_32x32_BN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),      # ✳️ 加 BN
            nn.ReLU(),
            nn.MaxPool2d(2),         # [64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),     # ✳️ 加 BN
            nn.ReLU(),
            nn.MaxPool2d(2),         # [128, 8, 8]

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),     # ✳️ 加 BN
            nn.ReLU(),
            nn.MaxPool2d(2),         # [128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),          # 调整 dropout 不要过强
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# class MNISTModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 200)
#         self.fc2 = nn.Linear(200, 10)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = self.dropout(torch.relu(self.fc1(x)))
#         return self.fc2(x)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 修改为 64 * 5 * 5 = 1600
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Output: (32, 32, 13, 13)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Output: (32, 64, 5, 5)
        x = x.view(x.size(0), -1)  # 自动展平为 1600
        x = F.relu(self.fc1(x))  # 输入到全连接层
        x = self.fc2(x)  # 输出
        return x



class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 增加一层，提升表达能力
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
