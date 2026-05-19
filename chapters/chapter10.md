# 第十章 集成学习

## 10.1 集成学习概述

集成学习通过组合多个弱学习器来构建强学习器，有效提升模型泛化能力。

**核心思想：** "集思广益，博采众长"

## 10.2 Bagging

**基本原理：** Bagging是一族并行化集成学习算法，通过训练多个独立的基学习器h₁,…,h_T，将它们的预测结果进行平均或投票，以得到最优结果。

**关键特点：**
- **自助采样：** 对样本的抽取是有放回的，因此理论上会有部分样本可能一直没有被选中
- **并行训练：** 各个基学习器可以并行训练
- **降低方差：** 通过集成多个模型的预测，降低模型的方差，提高泛化能力

**实践案例：** 基于Bagging的分类器集成

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging分类器
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
```

## 10.3 Boosting

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# Gradient Boosting
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbc.fit(X_train, y_train)
```

## 10.4 Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 定义基学习器列表
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

# Stacking分类器
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
```

## 10.5 XGBoost

```python
import xgboost as xgb

# XGBoost分类器
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    objective='binary:logistic',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# 特征重要性
xgb.plot_importance(xgb_model)
plt.show()
```

## 10.6 LightGBM

**基本原理：** 同样基于梯度提升框架，通过迭代地训练决策树并将它们组合起来，利用前向分步加法模型，每次在已有模型的基础上添加一个新的弱学习器来纠正之前的错误。

**核心特性：**
- **直方图算法：** 将连续的特征值离散化为有限个区间，构建直方图来统计特征的分布信息，大大减少了计算量
- **Leaf-wise生长策略：** 每次选择增益最大的叶子节点进行分裂，能够在更短的时间内找到更优的解
- **数据并行与特征并行：** 支持两种并行计算方式，提高训练速度
- **单边梯度采样：** 根据梯度的绝对值对数据进行排序，保留梯度较大的数据，减少计算量
- **互斥特征捆绑：** 将互斥的特征捆绑在一起，减少特征维度

**实践案例：** 基于LightGBM的大规模数据分类

```python
import lightgbm as lgb

# LightGBM分类器
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    objective='binary',
    random_state=42
)
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)

# 特征重要性
lgb.plot_importance(lgb_model)
plt.show()
```

---

[返回目录](../README.md)
