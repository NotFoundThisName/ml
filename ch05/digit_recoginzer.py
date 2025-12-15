import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #划分数据集
from sklearn.linear_model import LinearRegression, LogisticRegression  # 逻辑回归
from sklearn.preprocessing import MinMaxScaler #归一化 0附近和255附近的数字比较多，和正态分布反着来的，所以不用标准化

#1.加载数据集
dataset = pd.read_csv('../data/train.csv')

#2. 划分数据集
X = dataset.drop(columns=['label'],axis=1)
y = dataset['label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# digit = x_train.iloc[10].values.reshape(28,28)
# plt.imshow(digit , cmap='gray')
# plt.show()

#3.特征工程 归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#4.定义模型
model = LogisticRegression(max_iter=500)

#5.训练模型
model.fit(x_train, y_train)

#6.训练模型
accuracy = model.score(x_test, y_test)
print(accuracy)

#7.预测
digit = x_test[123]
pred = model.predict(digit.reshape(1, -1))
print(pred)
print(y_test.iloc[123])
plt.imshow(digit.reshape(28,28) , cmap='gray')
plt.show()