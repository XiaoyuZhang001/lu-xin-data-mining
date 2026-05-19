# 第五章 降维

## 5.1 降维概述

降维是将高维数据映射到低维空间的过程，目的是：
- 减少计算复杂度
- 去除冗余信息
- 可视化高维数据
- 缓解"维度灾难"

## 5.2 奇异值分解（SVD）

```python
from scipy.linalg import svd

# 奇异值分解
U, S, Vt = svd(X)

# 选择前k个奇异值
k = 2
X_reduced = U[:, :k] @ np.diag(S[:k])

print(f"原始维度: {X.shape}")
print(f"降维后维度: {X_reduced.shape}")
```

**图像压缩应用：**

```python
from PIL import Image

# 加载图像
img = Image.open('image.jpg').convert('L')
img_array = np.array(img)

# SVD压缩
U, S, Vt = svd(img_array)

# 保留前50个奇异值
k = 50
compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# 显示压缩前后对比
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img_array, cmap='gray')
plt.title('原始图像')
plt.subplot(122)
plt.imshow(compressed, cmap='gray')
plt.title(f'压缩图像 (保留{k}个奇异值)')
plt.show()
```

## 5.3 主成分分析（PCA）

```python
from sklearn.decomposition import PCA

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA降维结果')
plt.colorbar()
plt.show()
```

## 5.4 因子分析

```python
from sklearn.decomposition import FactorAnalysis

# 因子分析
fa = FactorAnalysis(n_components=3)
X_fa = fa.fit_transform(X)

print(f"因子载荷矩阵:\n{fa.components_}")
```

## 5.5 多维尺度变换（MDS）

```python
from sklearn.manifold import MDS

# MDS降维
mds = MDS(n_components=2, dissimilarity='euclidean')
X_mds = mds.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap='viridis')
plt.title('MDS降维结果')
plt.show()
```

## 5.6 等距特征映射（ISOMAP）

```python
from sklearn.manifold import Isomap

# ISOMAP降维（保留流形结构）
isomap = Isomap(n_components=2, n_neighbors=5)
X_isomap = isomap.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap='viridis')
plt.title('ISOMAP降维结果')
plt.show()
```

## 5.7 线性判别分析（LDA）

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA降维（有监督，最大化类间距离）
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.title('LDA降维结果')
plt.show()
```

## 5.8 t-SNE

```python
from sklearn.manifold import TSNE

# t-SNE降维（非线性，适合可视化）
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE降维结果')
plt.show()
```

## 5.9 UMAP

```python
import umap

# UMAP降维（保持全局结构）
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.title('UMAP降维结果')
plt.show()
```

---

[返回目录](../README.md)
