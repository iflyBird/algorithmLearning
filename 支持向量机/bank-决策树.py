from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv
from sklearn.model_selection import train_test_split
from sklearn import svm
# 读入数据
Dtree = open(r'bank_out', 'r')
reader = csv.reader(Dtree)

# 获取第一行数据
headers = reader.__next__()
print(headers)

# 定义两个列表
featureList = []
labelList = []

# 将CSV的文件存到featureList中
for row in reader:
    # 把label存入list
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        #建立一个数据字典
        rowDict[headers[i]] = row[i]
    # 把数据字典存入list
    featureList.append(rowDict)

#print(featureList)


# 把数据转换成01表示
#DictVectorizer的处理对象是符号化(非数字化)的但是具有一定结构的特征数据，
# 如字典等，将符号转成数字0/1表示
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()
# 打印属性名称
#print(vec.get_feature_names())

#print("x_data: ")
#print(x_data)

# 打印标签
#print("labelList: " + str(labelList))

# 把标签转换成01表示
#preprocessing.LabelBinarizer将标签二值化
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
#print("y_data: " )
#print(y_data)

x_train,x_test, y_train, y_test  = train_test_split(x_data,y_data,test_size=0.3, random_state=0)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

# 创建决策树模型
model = tree.DecisionTreeClassifier(criterion="entropy")
# 输入数据建立模型
print('****************decision tree train begin**************')
model.fit(x_train, y_train)
print('****************decision tree train end**************')
'''
# 创建svm模型
model = svm.SVC(gamma='scale')
# 输入数据建立模型
print('****************svm train begin**************')
model.fit(x_train, y_train)
print('****************svm train end**************')
'''
"""
# 测试
x_test = x_data[-3]
#print("x_test: " + str(x_test))

#reshape(1,-1)将数据变成一行，reshape(-1,1)将数据变成一列
predict = model.predict(x_test.reshape(1,-1))
print("predict: " + str(predict))
"""

print('****************predict begin**************')
i = 0
right_sum = 0
while i<len(x_test):
    predict = model.predict(x_test[i].reshape(1,-1))
    if predict == y_test[i]:
        right_sum += 1
    i += 1

accuracy = right_sum/len(y_test)
print("accuracy:{0}".format(accuracy))
print('****************predict end**************')


# 导出决策树
# pip install graphviz
# http://www.graphviz.org/
"""
import graphviz

dot_data = tree.export_graphviz(model,
                                out_file = None,
                                feature_names = vec.get_feature_names(),
                                class_names = lb.classes_,
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph.render('computer')

#表格第一行是判断条件，左边是TURE，右边是FALSE
#第二行是熵
#samples该节点的样本总数
#value[5,9],表示no的个数是5个，yes的个数是9个
#class是改节点的判断结果
graph

vec.get_feature_names()

lb.classes_
"""

