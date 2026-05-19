# 第七章 分类

## 7.1 分类概述

**分类的定义：** 对于数据集D={x₁,…,xₙ}，其中的每一个样本x=(x₁,…,x_d;c)均有d(d≥1)个特征，以及与之对应的类别标签c。分类便是根据数据特征对事物所属类别进行判断的数据挖掘技术。

**分类的应用场景：**
- 垃圾邮件识别
- 医疗诊断
- 信用风险评估
- 图像识别

## 7.2 评价指标

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# 计算评价指标
y_true = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1]

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵:")
print(cm)

# 准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"准确率: {accuracy:.4f}")

# 精准率
precision = precision_score(y_true, y_pred)
print(f"精准率: {precision:.4f}")

# 召回率
recall = recall_score(y_true, y_pred)
print(f"召回率: {recall:.4f}")

# F1得分
f1 = f1_score(y_true, y_pred)
print(f"F1得分: {f1:.4f}")

# AUC
auc = roc_auc_score(y_true, [0.1, 0.9, 0.6, 0.8, 0.2, 0.4, 0.1, 0.9])
print(f"AUC: {auc:.4f}")

# MCC
mcc = matthews_corrcoef(y_true, y_pred)
print(f"马修斯相关系数: {mcc:.4f}")
```

## 7.3 K-近邻（KNN）

```python
from sklearn.neighbors import KNeighborsClassifier

# KNN分类
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"KNN准确率: {accuracy_score(y_test, y_pred):.4f}")
```

## 7.4 朴素贝叶斯

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# 高斯朴素贝叶斯（连续特征）
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# 多项式朴素贝叶斯（离散特征）
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
```

## 7.5 决策树

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 决策树分类
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# 可视化决策树
plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=['Age', 'Income', 'Score'], class_names=['No', 'Yes'], filled=True)
plt.show()
```

## 7.6 支持向量机（SVM）

```python
from sklearn.svm import SVC

# 线性SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# 非线性SVM（RBF核）
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
```

## 7.7 人工神经网络

**基本原理：** 通过数学模型来模仿大脑神经网络的模型。深度神经网络通过引入更多的隐藏层和参数，能更有效地学习和表示复杂的数据模式，从而提高其在大规模数据和复杂任务下的性能表现。

**网络结构：**
- **输入层：** 接收原始数据
- **隐藏层：** 提取特征
- **输出层：** 输出预测结果

**应用场景：**
- 基于面部特征的性别分类
- 图像识别
- 语音识别

```python
from sklearn.neural_network import MLPClassifier

# 多层感知器
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', 
                    solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(f"MLP准确率: {accuracy_score(y_test, y_pred):.4f}")
```

---

[返回目录](../README.md)
