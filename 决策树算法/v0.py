#计算香农熵
import numpy as np
# def calEnt(dateset):
#     n=dateset.shape[0];#总行数
#     iset=dateset.iloc[:,-1].value_counts()#获取种类
#     p=iset/n#计算每个种类的概率
#     ent=(-p*np.log2(p)).sum();#计算熵
#     return ent;#返回熵的值
# #建立数据
from pandas import read_csv
from pandas import DataFrame
import math
database=DataFrame(read_csv(r'bank-full.csv',sep=';'))
#flag="yes"
result=[]

for i in range(database.shape[0]):
    #所有值放入列表中
    result.append(database.iloc[i,6])

print(result)
for i in range(len(result)):
    if(result[i]=="yes"):
        result[i]=1
    else:
        result[i]=0
print(result)
#result = list(map(int, result))
#print(result)
# for i, v in enumerate(result):
#
#     result[i] = int(v)
# print(type(result))



    #database.iloc(i, 6)==int(database.iloc(i, 6))
    #print(database.iloc(i,0))
    #for j, v in enumerate(numbers): numbers[j] = int(v)

    # numbers = [int(m) for m in numbers]
    # print(numbers)
    # if(database.iloc(i,6)=="yes"):da
    #
    #    database.iloc(i, 6)==map(int,database.iloc(i, 6))
    #    database.iloc(i, 6)==1
    # else:
    #     database.iloc(i, 6) == map(int, database.iloc(i, 6))
    #     database.iloc(i, 6) == 0
