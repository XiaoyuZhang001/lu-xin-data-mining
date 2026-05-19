# 第七章 分类

## 7.1 分类概述

分类是监督学习任务，目标是根据特征将样本分配到预定义的类别中。

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
