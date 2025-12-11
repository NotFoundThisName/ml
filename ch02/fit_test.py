# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #划分测试集和训练集
from sklearn.preprocessing import PolynomialFeatures #构建多项式特征
from sklearn.linear_model import LinearRegression #线性回归模型
from sklearn.metrics import mean_squared_error #均方误差

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] =False

#1.生成数据
X= np.linspace(-3,3,300).reshape(-1,1) #将数据生成为(300,1)
y = np.sin(X) + np.random.uniform(low=-0.5,high=0.5,size=300).reshape(-1,1)

print(X.shape)
print(y.shape)

#画出散点图
fig, ax = plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(X,y,color='red')
ax[1].scatter(X,y,color='red')
ax[2].scatter(X,y,color='red')
plt.tight_layout()
#plt.show()

#2.划分数据集
x_train , x_test, y_train , y_test =  train_test_split(X , y , test_size=0.2, random_state=42) #数据集X y 测试集的比例0.2 随机数种子42

#3. 定义模型 线性回归模型
model = LinearRegression()

#4. 分三种情况进行训练和测试
#4.1 欠拟合（一条直线）
x_train1 = x_train
x_test1 = x_test
model.fit(x_train1,y_train)

#查看训练完成后的模型参数
print(model.coef_)
print(model.intercept_)

#画出拟合直线
ax[0].scatter(X,model.predict(X),color='green')

#测试
y_pred1= model.predict(x_test1)

#计算误差 训练误差和测试误差
test_error= mean_squared_error(y_test,y_pred1)
train_error= mean_squared_error(y_train,model.predict(x_train1))

ax[0].text(-3,1,f"测试误差：{test_error:.4f}")
ax[0].text(-3,1.3,f"训练误差：{train_error:.4f}")

#4.2 恰好拟合 （5次训练）
ploy5 = PolynomialFeatures(degree=5) #5是最高的阶数

x_train2 = ploy5.fit_transform(x_train)
x_test2 = ploy5.transform(x_test)
model.fit(x_train2,y_train)

#查看训练完成后的模型参数
print(model.coef_)
print(model.intercept_)

#画出拟合直线
ax[1].scatter(X,model.predict(ploy5.fit_transform(X)),color='green')

#测试
y_pred2= model.predict(x_test2)

#计算误差 训练误差和测试误差
test_error2= mean_squared_error(y_test,y_pred2)
train_error2= mean_squared_error(y_train,model.predict(x_train2))

ax[1].text(-3,1,f"测试误差：{test_error2:.4f}")
ax[1].text(-3,1.3,f"训练误差：{train_error2:.4f}")

#4.3 过拟合 （20次训练）
ploy20 = PolynomialFeatures(degree=20) #20是最高的阶数

x_train3 = ploy20.fit_transform(x_train)
x_test3 = ploy20.transform(x_test)
model.fit(x_train3,y_train)

#查看训练完成后的模型参数
print(model.coef_)
print(model.intercept_)

#画出拟合直线
ax[2].scatter(X,model.predict(ploy20.fit_transform(X)),color='green')

#测试
y_pred3= model.predict(x_test3)

#计算误差 训练误差和测试误差
test_error3= mean_squared_error(y_test,y_pred3)
train_error3= mean_squared_error(y_train,model.predict(x_train3))

ax[2].text(-3,1,f"测试误差：{test_error3:.4f}")
ax[2].text(-3,1.3,f"训练误差：{train_error3:.4f}")


plt.show()