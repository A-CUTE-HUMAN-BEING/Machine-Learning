#SVM鸢尾花分类

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# Scikit-learn库没有提供拉普拉斯核函数,所以自定义拉普拉斯核函数
def laplacian_kernel(X, Y):
    sigma = 1  # 控制核宽度的参数
    kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            kernel_matrix[i, j] = np.exp(-np.linalg.norm(x - y, ord=1) / sigma)
    return kernel_matrix

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]  # 只选择前两个特征
y = iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义不同的惩罚参数
C_v = [0.1, 1, 10]

# 用不同的惩罚参数训练SVM模型
results = {}
for C in C_v:
    # 创建SVM模型,不使用默认的核函数
    svm_model = SVC(kernel='precomputed', C=C)
    # 计算核矩阵
    kernel_train = laplacian_kernel(X_train, X_train)
    # 在训练集上训练模型
    svm_model.fit(kernel_train, y_train)
    # 计算测试集的核矩阵
    kernel_test = laplacian_kernel(X_test, X_train)
    # 在测试集上进行预测
    y_pred = svm_model.predict(kernel_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 计算混淆矩阵为了计算精确率、召回率
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 计算精度和召回率（对于多分类问题，precision和recall采用micro平均）
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    # 存储结果
    results[C] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}

# 打印结果
print("不同惩罚参数下的模型分类效果:")
for C, metrics in results.items():
    print("惩罚参数: {}, 准确率: {:.2f}, 精度: {:.2f}, 召回率: {:.2f}".format(C, metrics['Accuracy'], metrics['Precision'], metrics['Recall']))

# 惩罚函数为1时,模型较好
svm_model = SVC(kernel='precomputed', C=1)
kernel_train = laplacian_kernel(X_train, X_train)
svm_model.fit(kernel_train, y_train)

# 绘制决策边界
def plot_SVM(X, y, model, title):
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(laplacian_kernel(np.c_[xx.ravel(), yy.ravel()], X_train))
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title(title)
    plt.show()

plot_SVM(X_train, y_train, svm_model, "SVM in iris data")

# 绘制ROC曲线
def plot_ROC(X, y, model):
    kernel_test = laplacian_kernel(X, X_train)
    y_proba = svm_model.decision_function(kernel_test)
    y_binary = label_binarize(y, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], y_proba[:, i])
        roc_auc[i] = roc_auc_score(y_binary[:, i], y_proba[:, i])

    colors = ['blue', 'green', 'red']
    penalties = [0.1, 1, 10]
    for i, color, penalty in zip(range(3), colors, penalties):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='C = {} (Area  = {:.2f})'.format(penalty, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.title('ROC Curve')
    #放右下角
    plt.legend(loc="lower right")
    plt.show()


plot_ROC(X_test, y_test, svm_model)