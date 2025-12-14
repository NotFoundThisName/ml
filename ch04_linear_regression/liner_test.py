import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #标准化
from sklearn.metrics import mean_squared_error #均方误差

from ch02_base.classification_report import y_pred

#1. 加载数据集
data = pd.read_csv('../data/advertising.csv')

data.drop(data.columns[0],axis=1, inplace=True)
data.dropna(inplace=True)

#2.划分数据集
X = data.drop(columns=['Sales'] ,axis=1)
y = data['Sales']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#3. 特征工程 标准化处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#4.定义模型
model_lr = LinearRegression()
model_sgd = SGDRegressor()

#5.训练模型
model_lr.fit(x_train, y_train)
print("正规方程法模型系数/斜率",model_lr.coef_)
print("正规方程法模型截距",model_lr.intercept_)

model_sgd.fit(x_train, y_train)
print("SGD模型系数/斜率",model_sgd.coef_)
print("SGD模型截距",model_sgd.intercept_)

#6.验证
y_pred1 = model_lr.predict(x_test)
y_pred2 = model_sgd.predict(x_test)

#7.使用评价指标来评价模型
print("正规方程法 MSE",mean_squared_error(y_test, y_pred1))
print("SGD MSE",mean_squared_error(y_test, y_pred2))

print("正规方程法决定系数" , model_lr.score(x_test, y_test))
print("SGD决定系数" , model_sgd.score(x_test, y_test))
