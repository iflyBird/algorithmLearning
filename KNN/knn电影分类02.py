#导入包
import pandas as pd
#设置数据集
#字典
rowdate={'电影名称':['无问东西','后来的我们','前任3','红海行动','唐人街探案','战狼'],
         '打斗镜头':[1,5,12,108,112,115],
         '接吻镜头':[101,89,97,5,9,8],
         '电影类型':['爱情片','爱情片','爱情片','动作片','动作片','动作片']}
movie_date=pd.DataFrame(rowdate)#将数据集转换为pd模式
# print(rowdate);
# print(movie_date);
#传入新的数据
new_date=[24,67]
#计算离每个电影的距离
dist=list((((movie_date.iloc[:6,1:3])-new_date)**2).sum(1)**0.5)
#print(dist);
#将计算好的距离排序
dist_1=pd.DataFrame({'dist':dist,'lables':(movie_date.iloc[:6,3])})
#print(dist_1);
dr=dist_1.sort_values(by='dist')[:4]#找到距离最近的四个
#print(dr)
#确定前4个所出现的频率
re=dr.loc[:,'lables'].value_counts()
print(re)
result=[]
result.append(re.index[0])
print(result)
# #封装方法
# import pandas as pd
# def classif0(inx,dateset,k):
#     result=[];
#     dist = list((((dateset.iloc[:6, 1:3]) - inx) ** 2).sum(1) ** 0.5)
#     dist_1 = pd.DataFrame({'dist': dist, 'lables': (dateset.iloc[:6, 3])})
#     dr = dist_1.sort_values(by='dist')[:k]
#     re=dr.loc[:,'lables'].value_counts();
#     result.append(re.index[0]);
#     return result;
# new_date=[24,67];
# inx=new_date
# rowdate={'电影名称':['无问东西','后来的我们','前任3','红海行动','唐人街探案','战狼'],
#           '打斗镜头':[1,5,12,108,112,115],
#           '接吻镜头':[101,89,97,5,9,8],
#           '电影类型':['爱情片','爱情片','爱情片','动作片','动作片','动作片']}
# dateset=rowdate;
# k=3;
# classif0(inx,dateset,k)