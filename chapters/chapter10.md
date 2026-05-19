# 第十章 集成学习

## 10.1 集成学习概述

集成学习通过组合多个弱学习器来构建强学习器，有效提升模型泛化能力。

**核心思想：** "集思广益，博采众长"

## 10.2 Bagging

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
