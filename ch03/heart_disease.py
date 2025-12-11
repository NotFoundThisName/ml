import pandas as pd
from pydantic.experimental.pipeline import transform
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.model_selection import GridSearchCV

#1 加载数据集
dataset = pd.read_csv('../data/heart_disease.csv')
print(dataset.head())

#处理缺失值
dataset.dropna(inplace=True)

#2. 数据集划分
X= dataset.drop(['是否患有心脏病'], axis=1)
y = dataset["是否患有心脏病"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 特征工程:特征转换
# 数值型特征
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]

#创建列转换器
column_transformer = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical_features),
        ('cat' , OneHotEncoder(drop="first"),categorical_features),
        ('bin',"passthrough",binary_features)
])

#执行特征转换
x_train = column_transformer.fit_transform(x_train)
x_test = column_transformer.transform(x_test)

# #4.定义模型
# knn = KNeighborsClassifier(n_neighbors=3)
#
# #5.训练模型
# knn.fit(x_train, y_train)
#
# #6.模型评估
# score = knn.score(x_test, y_test)
#
# print(score)

#9.网格搜索交叉验证
#创建基础分类器
knn = KNeighborsClassifier()
#定义参数网格
params_grid = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance'] , 'p' : [1 , 2]}

gs_cv = GridSearchCV(estimator=knn, param_grid=params_grid,cv=10)

#模型训练以及评估
gs_cv.fit(x_train, y_train)

#输出验证结果
print(gs_cv.best_estimator_)
print(gs_cv.best_params_)
print(gs_cv.best_score_)

#使用最优模型测试
knn = gs_cv.best_estimator_
print(knn.score(x_test, y_test))