import random


"""
函数功能：随机切分训练集和测试集  参数说明：
dataSet:输入的数据集  rate：训练集所占比例
返回：切分好的训练集和测试集
"""
import numpy as np
import pandas as pd
import random
#对无表头的数据，则需设置header=None，否则第一行数据被作为表头：
#对有表头的数据设置header=None则会报错：
dataSet =pd.read_csv('bank-full.csv',header = None)
#打印前五条数据
print(dataSet.head())
print(dataSet.shape)
#切割数据集的函数
def randSplit(dataSet, rate):
    l = list(dataSet.index) #提取出索引
    random.shuffle(l) #随机打乱索引
    #print(random.shuffle(l))
    dataSet.index = l #将打乱后的索引重新赋值给原数据集
    n = dataSet.shape[0] #总行数
    m = int(n * rate) #训练集的数量
    train = dataSet.loc[range(m), :] #提取前m个记录作为训练集
    test = dataSet.loc[range(m, n), :] #剩下的作为测试集
    dataSet.index = range(dataSet.shape[0]) #更新原数据集的索引
    test.index = range(test.shape[0]) #更新测试集的索引
    return train, test
l=list(dataSet.index)
print(l)
random.shuffle(l)
dataSet.index = l
n = dataSet.shape[0] #总行数
m = int(n * 0.8) #训练集的数量
train = dataSet.loc[range(m), :]
print(train)
test = dataSet.loc[range(m, n), :]
test.index = range(test.shape[0])
print(test)
print("*******************")
print("打印出测试集")
#运行train和test
train,test=randSplit(dataSet,0.8)
print(test)

#测试模型效果
def gnb_classify(train,test):
    labels = train.iloc[:,-1].value_counts().index #提取训练集的标签种类
    mean =[] #存放每个类别的均值
    std =[] #存放每个类别的方差
    result = [] #存放测试集的预测结果
    for i in labels:
        item = train.loc[train.iloc[:,-1]==i,:] #分别提取出每一种类别
        i=labels[0]
        m = item.iloc[:,:-1].mean() #当前类别的平均值
        s = np.sum((item.iloc[:,:-1]-m)**2)/(item.shape[0]) #当前类别的方差
        mean.append(m) #将当前类别的平均值追加至列表
        std.append(s) #将当前类别的方差追加至列表
    means = pd.DataFrame(mean,index=labels) #变成DF格式，索引为类标签
    stds = pd.DataFrame(std,index=labels) #变成DF格式，索引为类标签
    for j in range(test.shape[0]):
        iset = test.iloc[j,:-1].tolist() #当前测试实例
        iprob = np.exp(-1*(iset-means)**2/(stds*2))/(np.sqrt(2*np.pi*stds)) #正态分布公式
        prob = 1 #初始化当前实例总概率
        for k in range(test.shape[1]-1): #遍历每个特征
            prob *= iprob[k] #特征概率之积即为当前实例概率
            cla = prob.index[np.argmax(prob.values)] #返回最大概率的类别（得到最大值的索引，在得到标签

            # ）
        result.append(cla)
    test['predict']=result
    #倒数第一列为我们预测的，倒数第二列为原来的标签
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean() #计算预测准确率
    print(f'模型预测准确率为{acc}')
    return test
gnb_classify(train,test)
#预测(随机切分)
# for i in range(10):
#     train,test= randSplit(dataSet, 0.8)
#     gnb_classify(train,test)













