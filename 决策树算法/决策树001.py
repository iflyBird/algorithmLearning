#计算香农熵
import numpy as np
def calEnt(dateset):
    n=dateset.shape[0];#总行数
    iset=dateset.iloc[:,-1].value_counts()#获取种类
    p=iset/n;#计算每个种类的概率
    ent=(-p*np.log2(p)).sum();#计算熵
    return ent;#返回熵的值
#建立数据集(海洋生物为例)
import pandas as pd
def createDateset():
    row_date={
        'no-surfacing':[1,1,1,0,0],
        'flippers':[1,1,0,1,1],
        'fish':['yes','yes','no','no','no']
    }
    dateset=pd.DataFrame(row_date);
    return dateset;
#创建一个数据集
dataset=createDateset();
#print('数据集的香农熵为:%s'%(calEnt(dataset)));
#选择最优的列进行切分,返回最佳切分列的索引
def bestSplite(dataset):
    baseEnt=calEnt(dataset);#计算信息熵
    bestGain=0;#初始化信息增益
    asix=-1;#初始化最佳的切分列，标签类
    for i in range(dataset.shape[1]-1):#对列进行循环
        levels=dataset.iloc[:,i].value_counts().index;#获取当前列的所有值
        ents=0;#初始化子节点的信息熵
        for j in levels:#对每一列的每一个取值进行循环
            childset=dataset[dataset.iloc[:,i]==j]#对每个子节点的dataframe
            ent=calEnt(childset);#计算每一个节点的信息熵
            ents+=(childset.shape[0]/dataset.shape[0])*ent;#计算当前列的信息熵
       # print("信息增益为:%s"%ents)
        infoGain=baseEnt-ents;#计算当前列的信息增益
       # print("当前的信息增益为:%s"%infoGain)
        if infoGain>bestGain:
            bestGain=infoGain;
            asix=i;
    return asix;#返回切分列所在的索引
#按照给定的列切分数据集
#dataset原始的数据集 asix是指定的索引列
#value指定属性值
#测试信息增益
def mysplite(dataset,axis,value):
    col=dataset.columns[axis];#获取索引的列的特征名字
    redateset=dataset.loc[dataset[col]==value,:].drop(col,axis=1);#将该列去除
    return redateset;#返回去除后剩下的列
#构建树
def createTree(dataset):
    featlist=list(dataset.columns)#提取出所有的的列
    classlist=dataset.iloc[:,-1].value_counts()#获取所有的类标签
    #判断是否结束
    if classlist[0]==dataset.shape[0] or dataset.shape[1]==1:
        #如果所有标签是同一个类的话结束
        #如果只剩下最后一列的话结束
        return classlist.index[0]#如果是，返回类标签
    axis=bestSplite(dataset)#找到当前的最佳标签
    bestfeat=featlist[axis]#获取该索引的特征
    mytree={bestfeat:{}}#采用字典的形式存储树的信息
    del featlist[axis]#删除当前的特征
    valuelist=set(dataset.iloc[:,axis])#提取最佳切分列的所有的属性值
    for value in valuelist:
        #对每一个属性值递归建树
        mytree[bestfeat][value]=createTree(mysplite(dataset,axis,value))
    return mytree;#返回已经建好的树
#bestSplite(dataset);
mytree=createTree(dataset)
# print(mytree)
#决策树的保存
# np.save('myTree.npy',mytree);
# #将建好的树保存为myTree.npy格式，为了节省时间，建好后的树立马将其保存
# #后续使用直接调用即可
# #打开树,直接调用load()函数即可
# read_mytree=np.load('myTree.npy').item()
# #print(read_mytree)
# #建立分类的函数验证决策树
#iniputtree：已经生成的决策树
#lables:存储选择的最优特征的标签
#testcv:测试数据列表，顺序对应原数据集
def classify(inputTree,lables,testcv):
    firstStr=next(iter(inputTree))#获得第一个节点
    seconDict=inputTree[firstStr]#获取到下一个字典
    feateIndex=lables.index(firstStr)#获取第一个节点所咋IDE索引
    for key in seconDict.keys():
        if testcv[feateIndex]==key:
            if type(seconDict[key])==dict:
                classLabel=classify(seconDict[key],lables,testcv);
            else:
                classLabel=seconDict[key];
    return classLabel;

#测试准确率的函数
def acc_classfiy(train,test):
    inputTree=createTree(train);#根据训练集生成一个树
    lables=list(train.columns)#对数据集的列名称
    result=[];#记录结果
    for i in range(test.shape[0]):
        testvc=test.iloc[i,:-1];
        classLable=classify(inputTree,lables,testvc);
        result.append(classLable);
    test['predict']=result;
    acc=(test.iloc[:,-1]==test.iloc[:,-2]).mean();
    print(f"模型的准确率为{acc}")
    return test;
train=dataset;
#
test=dataset.iloc[:3,:];
acc_classfiy(train,test);




