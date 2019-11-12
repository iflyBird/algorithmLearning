import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#导入数据集
iris=pd.read_csv('iris.txt',header=None)
print(iris.head())
#iris.shape()
#构建距离计算函数
#函数功能:计算俩个数据集之间的欧氏距离
#输入:俩个array数据集
#返回:俩个数据集之间的欧氏距离
#
def distEclud(arrA,arrB):
    d=arrA-arrB
    dist=np.sum(np.power(d,2),axis=1)
    return dist
#计算3的平方
np.power(3,2)
#编写随机生成质心的函数，此处用Numpy。random.uniform()函数随机生成质心
#函数功能:随机生成质心
#参数说明:
#dataset：包含的数据集
# k：簇得个数
# 返回:
# data_cent:k个质心
def randCent(dataSet,k):
    n=dataSet.shape[1]
    data_min=dataSet.iloc[:,:n-1].min()
    data_max=dataSet.iloc[:,:n-1].max()
    #k行n-列的质心,不包含标签
    data_cent=np.random.uniform(data_min,data_max,(k,n-1))
    return data_cent
iris_cent=randCent(iris,3)
print(iris_cent)

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m, n = dataSet.shape
    m, n = dataSet.shape
    centroids = createCent(dataSet, k)
    clusterAssment = np.zeros((m, 3))#初始化为0,不停地迭代
    clusterAssment[:, 0] = np.inf#表示无穷大
    clusterAssment[:, 1: 3] = -1#第一列和第二列
    #连接函数,array变成dataframe形式,ignore_index=True自动拼接，忽略原来标签
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index=True)
    clusterChanged = True
    #while判断质心是否改变(质心和所属类别是一一对应的)
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            dist = distMeas(dataSet.iloc[i, :n - 1].values, centroids)
            result_set.iloc[i, n] = dist.min()#初始化最大值的那一列
            result_set.iloc[i, n + 1] = np.where(dist == dist.min())[0]#提取出来放到n+1列位置，就是质心位置
        clusterChanged = not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all()
        if clusterChanged:
            cent_df = result_set.groupby(n + 1).mean()#n+1为本次迭代所在的质心位置
            centroids = cent_df.iloc[:, :n - 1].values#更新质心
            result_set.iloc[:, -1] = result_set.iloc[:, -2]#最后一列更新为这一次的这一列
        return centroids, result_set
# dataSet=iris
# k=3
# centroids = randCent(dataSet, k)
# print(centroids)
# m,n=dataSet.shape
# clusterAssment = np.zeros((m, 3))  # 初始化为0,不停地迭代
# clusterAssment[:, 0] = np.inf  # 表示无穷大
# clusterAssment[:, 1: 3] = -1  # 第一列和第二列
# print(clusterAssment)
# result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index=True)
# print(result_set)
# dist=distEclud(dataSet.iloc[1,:n-1].values,centroids)
# print(dist)
# #寻找最小值对应的索引
# dist.min()
# m=np.where(dist==dist.min())[0]
# print(m)
# #a依次跟b中没一个元素比较，有一个不同返回false
# b=np.array([1,2,3,])
# a=2
# print((a==b).all())
iris_cent,iris_result=kMeans(iris,3)
print(iris_cent)
print(iris_result)
#测试分类结果
print(iris_result.iloc[:,-1]).value_counts()