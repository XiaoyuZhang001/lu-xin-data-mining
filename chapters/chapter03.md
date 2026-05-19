# 第三章 相关分析

## 3.1 相关分析概述

相关分析用于研究变量之间的线性或非线性关系程度，是数据挖掘中重要的探索性分析工具。

## 3.2 线性相关分析

**正态性检验：**

```python
import scipy.stats as stats

# 图示法：Q-Q图
stats.probplot(data, plot=plt)
plt.title('Q-Q Plot')
plt.show()

# 假设检验：Shapiro-Wilk检验
stat, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk检验: 统计量={stat:.4f}, p值={p_value:.4f}")
```

**Pearson相关系数：**

```python
# 计算Pearson相关系数及p值
pearson_corr, pearson_p = stats.pearsonr(df['Age'], df['Income'])
print(f"Pearson相关系数: {pearson_corr:.4f}, p值: {pearson_p:.4f}")
```

**Spearman秩相关系数：**

```python
# 适用于非正态分布或有序数据
spearman_corr, spearman_p = stats.spearmanr(df['Age'], df['Income'])
print(f"Spearman秩相关系数: {spearman_corr:.4f}, p值: {spearman_p:.4f}")
```

**Kendall等级相关系数：**

```python
# 适用于小样本或有序数据
kendall_corr, kendall_p = stats.kendalltau(df['Age'], df['Income'])
print(f"Kendall等级相关系数: {kendall_corr:.4f}, p值: {kendall_p:.4f}")
```

## 3.3 非线性相关分析

**互信息：**

```python
from sklearn.metrics import mutual_info_score

# 计算互信息（需离散化）
def mutual_information(x, y, bins=10):
    x_discrete = pd.cut(x, bins=bins, labels=False)
    y_discrete = pd.cut(y, bins=bins, labels=False)
    return mutual_info_score(x_discrete, y_discrete)

mi = mutual_information(df['Age'], df['Income'])
print(f"互信息: {mi:.4f}")
```

**最大信息系数（MIC）：**

```python
# 使用minepy库计算MIC
from minepy import MINE

mine = MINE()
mine.compute_score(df['Age'].values, df['Income'].values)
mic = mine.mic()
print(f"最大信息系数(MIC): {mic:.4f}")
```

## 3.4 偏相关分析

偏相关分析用于控制其他变量影响下，两个变量之间的净相关关系。

```python
import pingouin as pg

# 控制变量Z时，X和Y的偏相关
partial_corr = pg.partial_corr(data=df, x='Age', y='Income', covar=['Education'])
print(partial_corr)
```

## 3.5 距离相关分析

**距离度量方法：**

```python
from scipy.spatial.distance import euclidean, cityblock, cosine, mahalanobis
from scipy.stats import hamming

# 欧氏距离
euclid_dist = euclidean([1, 2, 3], [4, 5, 6])

# 曼哈顿距离
manhattan_dist = cityblock([1, 2, 3], [4, 5, 6])

# 切比雪夫距离
chebyshev_dist = max([abs(a-b) for a,b in zip([1,2,3], [4,5,6])])

# 闵可夫斯基距离（p=2为欧氏距离，p=1为曼哈顿距离）
from scipy.spatial.distance import minkowski
minkowski_dist = minkowski([1, 2, 3], [4, 5, 6], p=3)

# 余弦相似度
cos_sim = 1 - cosine([1, 2, 3], [4, 5, 6])

# 马氏距离（考虑数据分布）
cov_matrix = df[['Age', 'Income']].cov().values
mahalanobis_dist = mahalanobis([30, 50000], [40, 60000], np.linalg.inv(cov_matrix))

# 汉明距离（用于二进制数据）
hamming_dist = hamming([0, 1, 0, 1], [1, 1, 0, 0])

# 杰卡德系数（用于集合相似性）
from sklearn.metrics import jaccard_score
jaccard_sim = jaccard_score([0, 1, 0, 1], [1, 1, 0, 0])
```

---

[返回目录](../README.md)
