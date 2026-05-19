# 第六章 关联规则挖掘

## 6.1 关联规则挖掘流程

**步骤1：生成频繁项集**
- 找出支持度 ≥ min_sup 的项集

**步骤2：生成强关联规则**
- 使用频繁项集生成满足 min_sup 和 min_conf 的关联规则

**关键指标：**
- **支持度（Support）**: P(A ∩ B) - 项集同时出现的概率
- **置信度（Confidence）**: P(B|A) = P(A ∩ B) / P(A) - 规则的可信度
- **提升度（Lift）**: P(B|A) / P(B) - 规则的强度

## 6.2 Apriori算法

**基本原理：** 采用逐层搜索的迭代方法来生成频繁项集。

**生成迭代过程：**
- **连接步：** 将Lk与自身连接产生候选k+1项集的集合Ck+1
- **剪枝步：** 修剪Ck+1，去除包含非频繁子集的项集

**算法流程：**
1. 扫描事务数据库TDB，对项进行支持度计数，生成频繁1项集的集合L1
2. 对Lk使用连接步产生候选k+1项集的集合Ck+1，使用剪枝步修剪Ck+1
3. 遍历扫描TDB，对Ck+1进行支持度计数，找到频繁k+1项集的集合Lk+1
4. 重复步骤2、3，直到Ck或Lk为空

**实践案例：** 基于Apriori算法的超市商品购买记录关联分析

```python
from mlxtend.frequent_patterns import apriori, association_rules

# 准备事务数据
transactions = [
    ['牛奶', '面包', '鸡蛋'],
    ['牛奶', '面包'],
    ['牛奶', '鸡蛋'],
    ['面包', '鸡蛋'],
    ['牛奶', '面包', '鸡蛋', '酸奶']
]

# 转换为稀疏矩阵
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 生成频繁项集
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets)

# 生成关联规则
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

## 6.3 FP-Growth算法

```python
from mlxtend.frequent_patterns import fpgrowth

# FP-Growth算法（更高效）
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
print(rules)
```

## 6.4 Eclat算法

**基本原理：** 利用先验性质，由Lk链接生成Ck+1。Eclat算法采用垂直数据格式，记录每个项的事务ID列表。

**特点：**
- 基于垂直数据格式
- 使用交集操作计算支持度
- 适合处理密集数据集

**实践案例：** 基于Eclat算法的糖尿病症状关联分析

```python
# Eclat算法（基于垂直数据格式）
from mlxtend.frequent_patterns import eclat

frequent_itemsets = eclat(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets)
```

## 6.5 关联规则评估指标

```python
# 计算各种评估指标
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# 添加其他指标
rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])
rules['conviction'] = (1 - rules['consequent support']) / (1 - rules['confidence'])

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']])
```

---

[返回目录](../README.md)
