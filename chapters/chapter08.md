# 第八章 聚类

## 8.1 层次聚类

**基本原理：** 根据层次分解的顺序，层次聚类算法可分为自上向下的分裂型层次聚类算法和自下向上的凝聚型层次聚类算法。

**凝聚层次聚类算法流程：**
1. 将每个样本i看作一个聚类Ci
2. 计算Ci和Cj之间的最小距离d(Ci,Cj)=min(d(x,y))
3. 将距离最小的两个聚类Ci和Cj进行合并，作为一个新的聚类C={Ci,Cj}
4. 重复步骤2、3，直到所有类最后合并成一类

**距离度量方法：**
- 单链接（最小距离）
- 全链接（最大距离）
- 平均链接（平均距离）
- 沃德方法（最小化方差）

**实践案例：** 基于层次聚类的小麦种子聚类

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hierarchical.fit_predict(X)

# 绘制树状图
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('层次聚类树状图')
plt.show()
```

## 8.2 K-Means算法

```python
from sklearn.cluster import KMeans

# K-Means聚类
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X)

# 肘部法则选择k
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, 'bo-')
plt.xlabel('簇数k')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.show()
```

## 8.3 高斯混合模型（GMM）

```python
from sklearn.mixture import GaussianMixture

# GMM聚类（软聚类）
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)
probs = gmm.predict_proba(X)  # 概率矩阵

print("各样本属于各簇的概率:")
print(probs[:5])
```

## 8.4 DBSCAN

```python
from sklearn.cluster import DBSCAN

# DBSCAN聚类（基于密度）
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

print(f"簇的数量: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"噪声点数量: {list(labels).count(-1)}")
```

## 8.5 OPTICS

```python
from sklearn.cluster import OPTICS

# OPTICS聚类（改进的DBSCAN）
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
labels = optics.fit_predict(X)

# 可达距离图
plt.figure(figsize=(10, 6))
plt.plot(optics.reachability_[optics.ordering_], marker='.', linestyle='None')
plt.title('OPTICS可达距离图')
plt.show()
```

## 8.6 谱聚类

**基本原理：** 一种基于图的聚类方法，通过利用数据的谱（特征值）属性来进行聚类。该算法首先构建样本的相似性矩阵，然后创建代表样本相互关系的图，通过图的拉普拉斯矩阵进行特征分解，构建新的特征空间，最后使用传统聚类算法对特征空间进行聚类。

**算法步骤：**
1. 利用选定的相似度度量方法，构建数据集的相似性矩阵S
2. 基于相似矩阵，进一步构造邻接矩阵A和度矩阵D
3. 通过度矩阵D和邻接矩阵A，计算得到拉普拉斯矩阵L=D-A
4. 对拉普拉斯矩阵进行标准化处理D^(-1/2)LD^(1/2)
5. 计算标准化拉普拉斯矩阵的前k个最小特征值所对应的特征向量
6. 将特征向量进行标准化处理，组成n×k的特征矩阵F
7. 将F中的每一行作为k维的样本，应用K-means聚类算法进行聚类

**实践案例：** 基于谱聚类的股票数据聚类

```python
from sklearn.cluster import SpectralClustering

# 谱聚类
spectral = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
labels = spectral.fit_predict(X)
```

---

[返回目录](../README.md)
