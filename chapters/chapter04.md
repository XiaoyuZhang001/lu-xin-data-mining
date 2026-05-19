# 第四章 回归分析

## 4.1 线性回归

**一元线性回归：**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 准备数据
X = df[['Age']].values
y = df['Income'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"回归系数: {model.coef_[0]:.4f}")
print(f"截距: {model.intercept_:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

**多元线性回归：**

```python
# 多变量回归
X = df[['Age', 'Education', 'Experience']].values
y = df['Income'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("回归系数:", model.coef_)
print("截距:", model.intercept_)
```

## 4.2 多重共线性

**多重共线性的定义：** 当回归模型中的自变量之间存在高度线性相关时，就称存在多重共线性问题。

**多重共线性的影响：**
- 回归系数估计不稳定，方差增大
- 变量的显著性检验不可靠
- 模型解释困难

**方差膨胀因子（VIF）：**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 计算VIF
vif_data = pd.DataFrame()
vif_data["feature"] = ['Age', 'Education', 'Experience']
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif_data)

# VIF > 5 表示存在严重共线性
# VIF > 10 表示共线性非常严重
```

## 4.3 岭回归

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 使用交叉验证选择最优alpha
ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(ridge, param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"最优alpha: {grid.best_params_['alpha']}")
best_ridge = grid.best_estimator_
y_pred = best_ridge.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

## 4.4 LASSO回归

```python
from sklearn.linear_model import Lasso

# LASSO回归（L1正则化，可实现特征选择）
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

print("非零系数数量:", np.sum(lasso.coef_ != 0))
print("系数:", lasso.coef_)
```

## 4.5 非线性回归

```python
from sklearn.preprocessing import PolynomialFeatures

# 多项式回归
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_poly, y_train)

X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)
print(f"多项式回归R²: {r2_score(y_test, y_pred):.4f}")
```

---

[返回目录](../README.md)
