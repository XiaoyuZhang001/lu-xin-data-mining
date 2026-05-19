# 第二章 数据描述与统计指标

## 2.1 数据预处理

**数据质量要素：** 准确性、完整性、一致性、时效性、相关性

**数据清洗：**

```python
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('online_retail.csv')

# 缺失值处理
# 方法1：删除含缺失值的记录
df_clean = df.dropna()

# 方法2：均值填充
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 方法3：中位数填充（适用于偏态分布）
df['Income'] = df['Income'].fillna(df['Income'].median())

# 方法4：众数填充（适用于分类数据）
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])

# 方法5：线性插值（适用于时间序列）
df['Temperature'] = df['Temperature'].interpolate(method='linear')

# 噪声处理 - 分箱平滑
def bin_smoothing(data, bin_width):
    bins = np.arange(min(data), max(data) + bin_width, bin_width)
    digitized = np.digitize(data, bins)
    bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
    smoothed = np.array([bin_means[i-1] for i in digitized])
    return smoothed

# 离群点检测 - 3σ准则
def detect_outliers_zscore(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    outliers = np.abs(z_scores) > threshold
    return outliers
```

**数据集成：**
- 实体识别：解决不同数据源中同一实体的标识问题
- 冗余数据处理：去除重复记录、合并高相关属性

**数据规约：**
- **维度规约**：主成分分析（PCA）、属性子集选择
- **数量规约**：直方图、聚类、抽样（随机抽样、分层抽样、系统抽样）

**数据变换：**
- **数据规范化**：
  ```python
  # 最小-最大规范化
  def min_max_normalize(data):
      return (data - data.min()) / (data.max() - data.min())
  
  # Z-score标准化
  def z_score_normalize(data):
      return (data - data.mean()) / data.std()
  ```
- **连续数据离散化**：等宽分箱、等深分箱、聚类离散化
- **分类数据编码**：独热编码、标签编码、频率编码

## 2.2 数据属性类型

| 属性类型 | 描述 | 示例 |
|---------|------|------|
| **标称属性** | 无顺序关系的类别 | 性别、颜色、品牌 |
| **二元属性** | 只有两个值的标称属性 | 是/否、真/假 |
| **序数属性** | 有顺序关系的类别 | 学历（小学/中学/大学） |
| **数值属性** | 可度量的数值 | 身高、体重、收入 |
| **比率标度** | 有绝对零点的数值 | 长度、时间、金额 |
| **区间标度** | 无绝对零点的数值 | 温度（摄氏度） |
| **离散属性** | 有限或可数的值 | 产品数量 |
| **连续属性** | 无限不可数的值 | 时间、距离 |

## 2.3 数据描述性统计

**集中趋势度量：**

```python
import pandas as pd
import scipy.stats as stats

data = pd.Series([23, 28, 28, 35, 42, 45, 48, 52, 58, 65])

# 平均值
mean = data.mean()

# 中位数
median = data.median()

# 众数
mode = stats.mode(data)[0][0]

# 几何均值
geometric_mean = stats.gmean(data)

# 调和均值
harmonic_mean = stats.hmean(data)
```

**离散程度度量：**

```python
# 极差
range_val = data.max() - data.min()

# 方差
variance = data.var()

# 标准差
std_dev = data.std()

# 变异系数（相对离散程度）
cv = (std_dev / mean) * 100

# 四分位差
iqr = data.quantile(0.75) - data.quantile(0.25)

# 异众比率（针对分类数据）
mode_count = data.value_counts().max()
variation_ratio = 1 - (mode_count / len(data))
```

**分布形态度量：**

```python
# 偏态系数
skewness = data.skew()
# skewness > 0: 右偏分布
# skewness < 0: 左偏分布
# skewness ≈ 0: 对称分布

# 峰态系数
kurtosis = data.kurtosis()
# kurtosis > 0: 尖峰分布
# kurtosis < 0: 平峰分布
# kurtosis ≈ 0: 正态分布
```

## 2.4 数据可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")

# 条形图 - 类别分布
plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=df)
plt.title('商品类别分布')
plt.xticks(rotation=45)
plt.show()

# 饼图 - 占比分析
category_counts = df['Category'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
plt.title('商品类别占比')
plt.show()

# 直方图 - 数值分布
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=20, kde=True)
plt.title('商品价格分布')
plt.show()

# 箱线图 - 异常值检测
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Price', data=df)
plt.title('不同类别商品价格分布')
plt.xticks(rotation=45)
plt.show()

# 折线图 - 时间序列
plt.figure(figsize=(12, 6))
df['Date'] = pd.to_datetime(df['Date'])
daily_sales = df.groupby('Date')['Sales'].sum()
daily_sales.plot()
plt.title('每日销售额趋势')
plt.show()

# 散点图 - 相关性分析
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Income', data=df, hue='Gender')
plt.title('年龄与收入关系')
plt.show()

# 气泡图 - 三维关系展示
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Age', y='Income', size='SpendingScore', 
                data=df, alpha=0.6, sizes=(20, 200))
plt.title('年龄、收入与消费得分关系')
plt.show()

# 热力图 - 相关性矩阵
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('变量相关性热力图')
plt.show()
```

---

[返回目录](../README.md)
