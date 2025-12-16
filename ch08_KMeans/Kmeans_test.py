import os
os.environ['OMP_NUM_THREADS'] = '2'
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

#1.随机生成数据样本点
X , y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=2,random_state=42)

fig , ax = plt.subplots(2,figsize=(8,8))
ax[0].scatter(X[:,0], X[:,1], c=y,label="data",s=50)
ax[0].set_title("原始数据")

#2.定义模型
kmeans = KMeans(n_clusters=3)

#3.训练模型
kmeans.fit(X)

#获取聚类的簇中心
print(kmeans.cluster_centers_)

#4.预测，得到每个样本点的簇中心
y_pred = kmeans.predict(X)


ax[1].scatter(X[:,0], X[:,1], c=y_pred,label="聚类数据",s=50)
ax[1].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c="red",label="簇中心",s=100)
ax[1].set_title("聚类结果")
ax[1].legend()
plt.show()