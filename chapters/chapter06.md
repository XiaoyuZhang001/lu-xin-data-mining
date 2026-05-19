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
