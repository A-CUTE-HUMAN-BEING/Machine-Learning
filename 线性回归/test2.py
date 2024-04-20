import numpy as np
import pandas as pa
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

print("多元最小二乘线性回归：")
# 引入数据集
boston_data=pa.read_csv("C:\\Users\\86139\\Desktop\\boston_housing_data.csv")


# 筛选是否有空数据的样本，并删去
# print(boston_data.isnull().sum())
boston_data.dropna(inplace=True)

# 多元线性回归
x=boston_data[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT']]
y=boston_data['PRICE']

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# 将特征数据多个一维数组转换为二维数组
X_train=np.column_stack((x_train, np.ones(len(x_train))))
X_test=np.column_stack((x_test, np.ones(len(x_test))))

# 使用最小二乘法求解参数
W=inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

# 获取最优的 w 和 b
w=W[:-1]
b=W[-1]

print(f'w:{w}')
print(f'b:{b}\n')

# 使用测试集进行预测
y_pred=X_test.dot(W)

# 计算MAE进行性能评估
mae=np.mean(np.abs(y_pred - y_test))
print(f'MAE: {mae}')

# 计算MSE进行性能评估
mse=np.mean((y_pred - y_test) ** 2)
print(f'MSE: {mse}')

# 计算RMSE进行性能评估
rmse=np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}\n')

print("最小二乘线性回归模型可视化：")
# 选用rm和price两列数据
n=boston_data['RM']
m=boston_data['PRICE']

# 划分训练集和测试集
n_train, n_test, m_train, m_test = train_test_split(n, m, test_size=0.2, random_state=30)

# 将特征数据转换为二维数组
N_train=np.column_stack((n_train, np.ones(len(n_train))))

# 使用最小二乘法求解参数
W1=inv(N_train.T.dot(N_train)).dot(N_train.T).dot(m_train)

# 获取最优的 w 和 b
w=W1[0]
b=W1[1]

print(f'w:{w}')
print(f'b:{b}\n')

# RM、PRICE的散点图
plt.figure(figsize=(10,10))
plt.scatter(n_train,m_train, s=30, c='red',edgecolors='black')
X=np.linspace(min(n_train), max(n_train))
Y=w * X + b
# 画拟合直线
plt.plot(X, Y, color='BLUE')
plt.xlabel('RM')
plt.ylabel('PRICE')
plt.title('RM VS PRICE')
plt.show()


p=boston_data['RM']
q=boston_data['PRICE']

# 初始化参数
w1=0
b1=0

# 定义学习率
a=0.01

# 梯度下降算法
for i in range(100000):
    # 计算预测值
    q_pred=w1*p+b1

    # 更新参数,学习率*对平方误差成本函数对w/b求偏导
    w1-=a*((-2/len(x))*np.sum(p*(q-q_pred)))
    b1-=a*((-2/len(x))*np.sum(q-q_pred))

print("由于MSE值较高,使用梯度下降算法再建立一次模型，得到w、b数值近似，最小二乘模型没有大问题，问题可能出现在异常数据中")
print("梯度下降线性回归：")
print(f'w:{w1}')
print(f'b:{b1}\n')