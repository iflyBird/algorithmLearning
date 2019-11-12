#误差平方的计算和sse计算
#函数功能
#参数说明
#     dataSet：原始数据集
#     cluster:kmeans聚类算法
#     k：簇的个数
# 返回:误差平方和sse
#验证数据集
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#导入数据集
# iris=pd.read_csv('testSet.txt',header=None)
# print(iris.head())
# testSet=pd.read_table('testSet.txt',header=None)
# print(testSet.head)
iris=pd.read_csv('iris.txt',header=None)
testSet=pd.read_table('testSet.txt',header=None)
print(iris.head())
def distEclud(arrA,arrB):
    d=arrA-arrB
    dist=np.sum(np.power(d,2),axis=1)
    return dist
def randCent(dataSet,k):
    n=dataSet.shape[1]
    data_min=dataSet.iloc[:,:n-1].min()
    data_max=dataSet.iloc[:,:n-1].max()
    #k行n-列的质心,不包含标签
    data_cent=np.random.uniform(data_min,data_max,(k,n-1))
    return data_cent
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    #m, n = dataSet.shape
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

zd=pd.DataFrame(np.zeros(testSet.shape[0]).reshape(-1,1))
test_set=pd.concat([testSet,zd],axis=1,ignore_index=True)
print(test_set.head())
test_cent,test_cluster=kMeans(test_set,4)
print(test_cent)
print(test_cluster.head())
def kclearingCurve(dataSet,cluster=kMeans,k=10):
    n=dataSet.shape[1]
    SSE=[]
    for i in range(1,k):
        centroids,result_set=cluster(dataSet,i+1)
        SSE.append(result_set.iloc[:,n].sum())
    plt.plot(range(2,k+1),SSE,'--o')
    return SSE
print(kclearingCurve(iris))
print(kclearingCurve(test_set))