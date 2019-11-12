import csv
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import pandas as pd
from sklearn import preprocessing
from itertools import chain
file_src = "bank-full.csv"
file_out = "bank_out.csv"

'''
Dtree = open(file_src,'r')
reader = csv.reader(Dtree,delimiter=';')

#获取表头
header_src = reader.__next__()
#print(header)

# 定义两个列表
data_src = []

for row in reader:
    data_src.append(row)
#print(data)

test = pd.DataFrame(columns=header_src,data=data_src)

#index=false  不要索引
test.to_csv(file_out,index=False)

'''
featureList = []
labelList = []

Dtree = open(file_out,'r')
reader = csv.reader(Dtree)

#获取表头
header_out = reader.__next__()
#将CSV的文件存到featureList中
for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[header_out[i]] = row[i]
    featureList.append(rowDict)

print(featureList)



