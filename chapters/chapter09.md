# 第九章 异常检测

## 9.1 异常检测概述

异常检测（Outlier Detection）是识别数据集中与大部分数据显著不同的样本的过程。

**主要挑战：**
- 数据稀疏性：异常样本比例低
- 噪声干扰：区分噪声与真实异常
- 解释性：理解异常产生原因

## 9.2 统计方法

**3σ准则：** 假设数据服从正态分布，约99.7%的数据落在均值±3σ范围内，超出此范围的数据视为异常值。

**箱线图方法：** 使用四分位数范围（IQR）来识别异常值，通常将低于Q1-1.5×IQR或高于Q3+1.5×IQR的数据视为异常值。

**基于直方图的异常值得分：** 通过计算数据点在直方图中的频率来评估其异常程度。

**累积和法：** 用于检测序列数据中的异常变化。

**实践案例：** 基于箱线图的wiki网络流量异常检测

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

**向量自回归模型（VAR）：** 用于多变量时间序列的预测和异常检测。

**自回归差分移动平均模型（ARIMA）：** 用于单变量时间序列的预测和异常检测。

**LSTM异常检测：**
- 使用长短期记忆网络学习时间序列的模式
- 通过预测误差来识别异常
- 适用于复杂的时间序列数据

**LSTM异常检测步骤：**
1. 数据预处理：对数据进行标准化处理
2. 模型构建：构建LSTM模型，包括输入层、LSTM层和输出层
3. 训练模型：目标为最小化损失函数Loss
4. 判断异常：计算每个时间步t的预测误差error，设定阈值ε，若error(t)>ε，则标记时刻t为异常

**实践案例：** 基于LSTM的股票收盘价格异常检测

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
