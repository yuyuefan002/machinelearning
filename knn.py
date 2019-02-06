import csv
import roc
import math
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np


def ReadDataFromFile(filename):
    '''
    This function read data from csv file while contains truth and lambda value.
    then return those values in list

    :param filename: the csv file from which we can get the truth and lambda
    :return: the list holding each line
    '''
    with open(filename, newline='') as csvfile:
        FileData = []
        feature1_h0 = []
        feature2_h0 = []
        feature1_h1 = []
        feature2_h1 = []
        SpamReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in SpamReader:
            classnum, feature1, feature2 = row[0].split(',')
            FileData.append((float(feature1), float(feature2), classnum))
        return FileData

def GenerateGrid(FileData, GridNum):
    '''
    This function generate the test data
    :param FileData: used to generate the range
    :param GridNum: test data num
    :return: test dataset
    '''
    def GenerateArrange(FileData):
        import numpy as np
        x = []
        y = []
        for data in FileData:
            x.append(data[0])
            y.append(data[1])
        return min(x), max(x), min(y), max(y)
    import numpy as np
    minX, maxX, minY, maxY = GenerateArrange(FileData)
    xRange = maxX - minX
    yRange = maxY - minY
    x = np.linspace(minX-0.2*xRange, maxX+0.2*xRange, GridNum)
    y = np.linspace(minY-0.2*yRange, maxY+0.2*yRange, GridNum)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def runClassifier(TrainData, TestData, k):
    '''
    This function runs the kNN
    :param TrainData: Train dataset
    :param TestData: Test dataset
    :param k: k for kNN
    :return: result of lambda
    '''
    def GetNeigh(TrainData, testInstance, k):
        '''
        This function gets the nearest k neighs
        :param trainingSet:
        :param testInstance:
        :param k:
        :return:
        '''
        def Distance(pointA, pointB):
            '''
            Computer the Euclidean distance between 2 points
            :param pointA:
            :param pointB:
            :return: distance
            '''
            distance = 0
            for i in range(len(pointA) - 1):
                distance += pow((pointA[i] - pointB[i]), 2)
            return math.sqrt(distance)

        distances = []
        for i in range(len(TrainData)):
            dist = Distance(TrainData[i], testInstance)
            distances.append((TrainData[i], dist))
        distances.sort(key=itemgetter(1))
        neighs = []
        for i in range(k):
            neighs.append(distances[i][0])
        return neighs

    def GetLambda(neighs, k):
        '''
        Computer the lambda for one data point
        :param neighs: k nearest neighs
        :param k: k for kNN
        :return: lambda
        '''
        h1 = 0
        for neigh in neighs:
            if neigh[2] == '1':
                h1 += 1
        return h1 / k

    lambdaVs = []
    for data in TestData:
        xx, yy, classnum = data
        neighs = GetNeigh(TrainData, [xx, yy, classnum], k=k)
        lambdaV = GetLambda(neighs, k=k)
        lambdaVs.append(lambdaV)
    return lambdaVs



def GetResponse(neighs):
    majorityVotes = {}
    for neigh in neighs:
        classnum = neigh[-1]
        if classnum in majorityVotes:
            majorityVotes[classnum] += 1
        else:
            majorityVotes[classnum] = 1
    sortedmajorityVotes = sorted(majorityVotes.items(), key=itemgetter(1), reverse=True)
    return sortedmajorityVotes[0][0]


def plot(Data, xx, yy, lambdaVs, levels):
    from matplotlib.colors import ListedColormap
    plt.figure(figsize=(12, 7))
    plt.contourf(xx, yy, lambdaVs, levels=levels, vmin=0, vmax=1)
    plt.colorbar()
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(lambdaVs))])
    cs = plt.contour(xx, yy, lambdaVs, cmap=cmap, levels=[.5])
    cs.collections[0].set_label('Majority Vote Decision Boundary')
    color = ['yellow', 'blue', 'green', 'black']
    class0 = [list(), list()]
    class1 = [list(), list()]
    for Datum in Data:
        if Datum[2] == '0':
            class0[0].append(Datum[0])
            class0[1].append(Datum[1])
        elif Datum[2] == '1':
            class1[0].append(Datum[0])
            class1[1].append(Datum[1])
    plt.scatter(x=class0[0], y=class0[1], color=color[0], label='class 0')
    plt.scatter(x=class1[0], y=class1[1], color=color[1], label='class 1')
    plt.legend(loc='lower right')
    plt.show()

def kNNVisualization(filename, k):
    '''
    This function is for visualize knn performance
    :param filename: the dataset
    :param k: k for kNN
    :return:
    '''
    k = k
    FileData = ReadDataFromFile(filename)
    xx, yy = GenerateGrid(FileData, GridNum=251)
    TestData=[]
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            TestData.append((xx[i][j],yy[i][j],'0'))
    lambdaVs = runClassifier(FileData, TestData, k)
    lambdaVs = np.array(lambdaVs).reshape(xx.shape)
    plot(FileData, xx, yy, lambdaVs, levels=k)

def kNNRoc(filename, k):
    k = k
    FileData = ReadDataFromFile(filename)

    for data in FileData:
        lambdaVs = runClassifier(FileData, FileData, k)
    pass


# question 3a
#kNNVisualization('dataSetHorseshoes.csv', 1)
#kNNVisualization('dataSetHorseshoes.csv', 5)
#kNNVisualization('dataSetHorseshoes.csv', 31)
kNNVisualization('dataSetHorseshoes.csv', 91)
# question 3c
#FileData = ReadDataFromFile('dataSetHorseshoes.csv')
#kNNVisualization('dataSetHorseshoes.csv', int(len(FileData)/2-1))
#kNNVisualization('dataSetHorseshoes.csv', int(len(FileData)-1))
#question 4
#kNNRoc('dataSetHorseshoes.csv',1)