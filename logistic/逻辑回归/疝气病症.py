import pandas as pd
import numpy as np
#导入测试集和训练集
test=pd.read_table(r'horseColicTest.txt');
# print(test);
# print('-------------------------')
train=pd.read_table(r'horseColicTraining.txt');
# print(train);
#定义sigmod函数
def sigmod(Inx):
    s=1/(1+np.exp(-Inx));
    return s;
#定义一个分类函数
def classifly(inX,weight):
    p=sigmod(inX*weight);
    if p<0.5:
        return 0;
    else:
        return 1;
#构造标准化函数,减小迭代次数
def BiaoZhun(Imat):
    shuju=Imat.copy();
    mean_1=np.mean(shuju,axis=0);
    std_1=np.std(shuju,axis=0);
    imat=(shuju-mean_1)/std_1;
    return imat;

def SGD_LR(dataset,alpha=0.01,maxcycle=500):
    dataset=dataset.sample(maxcycle,replace=True);#smaple随机抽取若干行replace表示是否有放回的抽无，false表示不放回的抽取，
    #true表示有放回的抽取
    dataset.index=range(dataset.shape[0]);
    Xmat = np.mat(dataset.iloc[:, :-1].values);
    Ymat = np.mat(dataset.iloc[:, -1].values).T;
    Xmat = BiaoZhun(Xmat);
    m, n = Xmat.shape;
    # print(m,n)
    weight = np.zeros((n, 1));
    for i in range(m):
        grad = ((Xmat[i].T) * (Xmat[i] * weight - Ymat[i]));
        weight = weight - alpha * grad;
    return weight;
def DGB_LR(dataset,alpha=0.001,maxcycle=500):
    Xmat=np.mat(dataset.iloc[:,:-1].values);
    Ymat=np.mat(dataset.iloc[:,-1].values).T;
    Xmat=BiaoZhun(Xmat);
    m,n=Xmat.shape;
    # print(m,n)
    weight=np.zeros((n,1));
    for i in range(maxcycle):
        grad=((Xmat.T)*(Xmat*weight-Ymat))/m;
        weight=weight-alpha*grad;
    return weight;
#构建逻辑回归模型
def logisticAcc(train,test,methond,alpha=0.01,maxcle=5000):
    ws=methond(train, alpha=alpha, maxcycle=maxcle)
    xMat=np.mat(test.iloc[:,:-1].values);
    result=[];
    #对xmat进行标准化
    xMat = BiaoZhun(xMat);
    #对测试集中的每一个样本进行分类一个标签
    for inx in xMat:
        label=classifly(inx,ws)
        #得到的labels值要么是0要么是1,放到里面
        result.append(label);
    retest=test.copy();
    #增加一列为预测的那一列
    retest['result']=result;
    acc=(retest.iloc[:,-1]==retest.iloc[:,-2]).mean();
    print(f'模型的的准确率为{acc}')
    return retest;

logisticAcc(train,test,SGD_LR,alpha=0.01,maxcle=5000)

#构造批量梯度下降构建函数

print('-------------------------------')
logisticAcc(train,test,DGB_LR,alpha=0.01,maxcle=5000)