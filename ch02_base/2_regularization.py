import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 划分训练集和测试集
from sklearn.preprocessing import PolynomialFeatures    # 构建多项式特征
from sklearn.linear_model import LinearRegression, Lasso, Ridge   # 线性回归模型，Lasso回归，岭回归
from sklearn.metrics import mean_squared_error  # 均方误差

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 1. 构建数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)

# print(X.shape)
# print(y.shape)

# 画出散点图
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].scatter(X, y, color='y')
ax[0,1].scatter(X, y, color='y')
ax[0,2].scatter(X, y, color='y')

#2. 划分数据集
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#3. 准备数据  构建21维特征
poly20 = PolynomialFeatures(degree = 20)
x_train = poly20.fit_transform(x_train)
x_test = poly20.transform(x_test)

#4. 定义模型
#4.1 不加正则化 线性回归模型
model = LinearRegression()

#模型训练
model.fit(x_train, y_train)

#预测
y_pred1 = model.predict(x_test)

#计算测试误差
test_loss1 = mean_squared_error(y_test, y_pred1)
ax[0,0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
# 拟合曲线
ax[0,0].plot(X, model.predict( poly20.fit_transform(X) ), color='r')
# 系数的直方图
ax[1,0].bar(np.arange(21), model.coef_.reshape(-1))

#4.2 加L1正则化 LOSS回归模型
lasso = Lasso(alpha=0.01)

#模型训练
lasso.fit(x_train, y_train)

#预测
y_pred2 = lasso.predict(x_test)

#计算测试误差
test_loss2 = mean_squared_error(y_test, y_pred2)
ax[0,1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
# 拟合曲线
ax[0,1].plot(X, lasso.predict( poly20.fit_transform(X) ), color='r')
# 系数的直方图
ax[1,1].bar(np.arange(21), lasso.coef_.reshape(-1))

#4.3 加L2正则化 岭回归模型
ridge = Ridge(alpha=0.5)

#模型训练
ridge.fit(x_train, y_train)

#预测
y_pred3 = ridge.predict(x_test)

#计算测试误差
test_loss3 = mean_squared_error(y_test, y_pred3)
ax[0,2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
# 拟合曲线
ax[0,2].plot(X, ridge.predict( poly20.fit_transform(X) ), color='r')
# 系数的直方图
ax[1,2].bar(np.arange(21), ridge.coef_.reshape(-1))
plt.show()