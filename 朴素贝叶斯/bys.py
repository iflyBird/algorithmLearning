#导入包（使用）
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#导入数据集
from sklearn import datasets
iris=datasets.load_iris()
#得到整个数据集
print(iris)
#得到的data数据为array
print(iris.data)
#得到分类
print(iris.target)
#切分数据集
#random_state为随机数种子，保证每次切分相等
Xtrain, Xtest, ytrain, ytest = train_test_split(iris.data,
iris.target,  random_state=42)
#得到训练集长度
print(len(Xtrain))

#得到测试集的长度
print(len(Xtest))
#建模
#把训练集进行测试一下·
clf = GaussianNB()
text1=clf.fit(Xtrain, ytrain)
print(text1)

#在测试集上执行预测，proba 导出的是每个样本属于某类的概率
#predict是y_text类别
clf.predict(Xtest)
print(clf.predict(Xtest))
gailv=clf.predict_proba(Xtest)
#计算出每一个某类的概率
print(gailv)
#测试准确率
result=accuracy_score(ytest,clf.predict(Xtest))
#得出训练结果为97%
print(result)
