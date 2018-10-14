#!/usr/bin/env python
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt



def createDateset():
    group = np.array([[1.0, 1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    
    return group, labels
##k-近邻算法
def classify(test, datesets, labels, k):
    ###测试数据与数据库中的数据做差
    diffMat = test - datesets
    ###欧氏距离
    distances = np.linalg.norm(diffMat, ord=2, axis=1)
    ###对距离进行排序，升序，取下标
    sortedDisIndices = distances.argsort()
    classCount = {}
    ###确定前k个点所在类别的出现频率
    for i in range(k):
        voteLabel = labels[sortedDisIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    ###对前K个点所在类别的出现频率进行排序
    sortedClassCount = sorted(classCount.items(),key=lambda item:item[1],reverse=True)
    ###返回字典第一个值,即分类标签
    return sortedClassCount[0][0] 
# ###读取dating数据，并将其转变成矩阵
def file2matrix(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        ###获取数据的数量
        data = f.readlines()
        numbers = len(data)
        ###创建矩阵
        dataMat = np.zeros((numbers, 3))
        ###类别标签
        classLabels = []
        for i, line in enumerate(data):
            line = line.strip().split()
            dataMat[i,:] = line[0:3]
            classLabels.append(int(line[-1]))
        
        return dataMat, classLabels
###归一化数据，newvalue = (oldValue-min)/(max-min)
def autoNorm(datasets):
    ###最小值
    minVals = datasets.min(axis=0)
    ###最大值
    maxVals = datasets.max(axis=0) 
    ranges = maxVals - minVals
    normDatesets = (datasets - minVals) / ranges
    return normDatesets, ranges, minVals         
    
###利用分类器针对约会网站进行测试 
def datingClassTest():
    ###测试样本比例，选取10%
    holeRatio = 0.1
    dataMat, classLabels = file2matrix('datingTestSet2.txt')
    normDatasets,ranges,minVals = autoNorm(dataMat)
    m = normDatasets.shape[0]
    numTest = int(m * holeRatio)
    errorCount = 0.0
    for i in range(numTest):
        classifierRes = classify(normDatasets[i,:], normDatasets[numTest:m,:], classLabels[numTest:m] , 3)
        print("the classifier result is %d, the real answer is %d" %(classifierRes,classLabels[i]))
        if classifierRes != classLabels[i]:
            errorCount +=1
    print("the total error rate is: %f" %(errorCount/float(numTest)))




if __name__ == "__main__":
    datingClassTest()
    
  

    

