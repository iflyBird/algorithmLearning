import pandas as pd
import numpy as np
#导入数据集
dataset=pd.read_table(r'testSet.txt',header=None);
print(dataset.head());
print('------------------');
#添加标签
dataset.columns=['X1','X2','Lables'];
print(dataset.head());
#构造sigmod函数
def sigmod(Inx):
    s=1/(1+np.exp(-Inx));
    return s;
#构造标准化函数,减小迭代次数,数据收敛更快
def BiaoZhun(Imat):
    shuju=Imat.copy();
    mean_1=np.mean(shuju,axis=0);
    std_1=np.std(shuju,axis=0);
    imat=(shuju-mean_1)/std_1;
    return imat;
#构造批量梯度下降构建函数
#maxcycle为最大迭代次数
#利用批量数据下降法
def DGB_LR(dataset,alpha=0.001,maxcycle=500):
    #x-mat为除了最后一列，前面提取出来
    Xmat=np.mat(dataset.iloc[:,:-1].values);
    #将labels提取出来.t表示转置，变成列
    Ymat=np.mat(dataset.iloc[:,-1].values).T;
    Xmat=BiaoZhun(Xmat);
    #m,n表示特征的个数
    m,n=Xmat.shape;
    # print(m,n)
    weight=np.zeros((n,1));
    #开始迭代
    for i in range(maxcycle):
        grad=((Xmat.T)*(Xmat*weight-Ymat))/m;
        weight=weight-alpha*grad;
    return weight;
# print(DGB_LR(dataset,alpha=0.001,maxcycle=500))
#获得每个权重
ws=DGB_LR(dataset,alpha=0.001,maxcycle=500)
xMat=np.mat(dataset.iloc[:,:-1].values);
yMat=np.mat(dataset.iloc[:,-1].values).T;
xMat=BiaoZhun(xMat);
#xmat乘以权重，进行扁平化处理
# A=(xMat*ws).A.flatten();
# A=xMat*ws;
# print(A);
# print('---------------------------------------')
#矩阵扁平化，使其成为一个矩阵
# A=xMat*ws;
# print(A1)
#计算sigmod的值
# p=sigmod(A).A.flatten();
# for i,j in enumerate(p):#enumerate i表示索引，j表示值
#     if j<0.5:
#         p[i]=0;
#     else:
#         p[i]=1;
# print('------------------')
# print(p);
# train_error=(np.fabs(yMat.A.flatten()-p)).sum();
# train_rate=train_error/yMat.shape[0];
# print(f'模型的错误率为{train_rate}')
#封装成函数
def logisticAcc(dataset,method,alpha=0.01,maxcle=500):
    ws = method(dataset, alpha=alpha, maxcycle=maxcle)
    xMat = np.mat(dataset.iloc[:,:-1].values);
    yMat = np.mat(dataset.iloc[:,-1].values).T;
    xMat = BiaoZhun(xMat);
    p=sigmod(xMat*ws).A.flatten();
    for i, j in enumerate(p):  # enumerate i表示索引，j表示值
        if j < 0.5:
            p[i] = 0;
        else:
            p[i] = 1;
    train_error = (np.fabs(yMat.A.flatten() - p)).sum();
    train_rate = train_error/yMat.shape[0];
    return train_rate;
#错误率为0.04,正确率为0.96
print(logisticAcc(dataset,DGB_LR,alpha=0.01,maxcle=500));
#随机梯度下降算法SGD_LR
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
print("----------------------");
#准确率为0.96
print(logisticAcc(dataset,SGD_LR,alpha=0.01,maxcle=500));