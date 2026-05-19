# 第九章 异常检测

## 9.1 异常检测概述

异常检测（Outlier Detection）是识别数据集中与大部分数据显著不同的样本的过程。

**主要挑战：**
- 数据稀疏性：异常样本比例低
- 噪声干扰：区分噪声与真实异常
- 解释性：理解异常产生原因

## 9.2 统计方法

```python
# 3σ准则
def detect_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold

# 箱线图方法
def detect_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound)
```

## 9.3 基于空间分布的方法

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 孤立森林
iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
outliers = iforest.fit_predict(X)
# -1表示异常，1表示正常

# 局部异常因子（LOF）
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outliers = lof.fit_predict(X)
```

## 9.4 基于降维的方法

```python
from sklearn.decomposition import PCA

# PCA异常检测
pca = PCA(n_components=2)
X_reconstructed = pca.inverse_transform(pca.fit_transform(X))
reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)

# 基于重构误差识别异常
threshold = np.percentile(reconstruction_error, 95)
outliers = reconstruction_error > threshold
```

## 9.5 基于预测的方法

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# LSTM异常检测
def build_lstm_autoencoder(input_dim, timesteps):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, input_dim)))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_dim * timesteps))
    model.compile(optimizer='adam', loss='mse')
    return model

# 训练并检测异常
model = build_lstm_autoencoder(input_dim=1, timesteps=24)
model.fit(X_train, y_train, epochs=50, batch_size=32)
predictions = model.predict(X_test)
error = np.mean((X_test - predictions) ** 2, axis=1)
```

---

[返回目录](../README.md)
