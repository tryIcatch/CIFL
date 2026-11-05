from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# 数据
distance = {'P0': 13.6924, 'P1': 13.6856, 'P2': 4.5712, 'P3': 4.5798, 'P4': 4.5691, 'P5': 4.5732, 'P6': 4.5771, 'P7': 4.5756}
angle = {'P0': 170.3791, 'P1': 170.5094, 'P2': 9.4841, 'P3': 9.6156, 'P4': 9.4566, 'P5': 9.5201, 'P6': 9.5640, 'P7': 9.5253}

# 将距离和角度转化为 2D 数据
data = np.array([[distance[p], angle[p]] for p in distance])

# 可视化数据点
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.xlabel('Distance')
plt.ylabel('Angle')
plt.title('2D Data (Distance vs Angle)')
plt.show()

# 执行 DBSCAN 聚类
db = DBSCAN(eps=1.5, min_samples=2)
labels = db.fit_predict(data)

# 打印聚类结果
print("DBSCAN Labels:", labels)
