
def ReadDataFromFile(filename):
    '''
    This function read data from csv file while contains truth and lambda value.
    then return those values in list

    :param filename: the csv file from which we can get the truth and lambda
    :return: the list holding each line
    '''
    import csv
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


def DLRTClassifier(TrainData, TestData, k):
    '''
    This function runs the kNN
    :param TrainData: Train dataset
    :param TestData: Test dataset
    :param k: k for kNN
    :return: result of lambda
    '''
    import math
    from operator import itemgetter
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
    def GetNeigh(TrainData, testInstance, k):
        '''
        This function gets the nearest k neighs
        :param trainingSet:
        :param testInstance:
        :param k:
        :return:
        '''
        distances = []
        for i in range(len(TrainData)):
            dist = Distance(TrainData[i], testInstance)
            distances.append((TrainData[i], dist))
        distances.sort(key=itemgetter(1))
        neighs = []
        for i in range(len(distances)):
            neighs.append(distances[i][0])

        return neighs

    def GetLambda(neighs, data, k,):
        '''
        Computer the lambda for one data point
        :param neighs: k nearest neighs
        :param k: k for kNN
        :return: lambda
        '''
        from math import log
        n0 = 0
        n1 = 0
        for i in range(len(neighs)):
            if neighs[i][2] == '0':
                n0 += 1
            elif neighs[i][2] == '1':
                n1 += 1
            if n0 == k:
                kh0 = neighs[i]
            if n1 == k:
                kh1 = neighs[i]
        if n0<k or n1 <k:
            print(n0,n1)
        lambdaV = log(n0/n1)+2*(log(Distance(data, kh0))-log(Distance(data, kh1)))
        return lambdaV

    lambdaVs = []
    for data in TestData:
        xx, yy, classnum = data
        neighs= GetNeigh(TrainData, [xx, yy, classnum], k=k)
        lambdaV = GetLambda(neighs, data, k=k, )
        lambdaVs.append(lambdaV)
    return lambdaVs


def DLRTVisualization(FileData, k):
    '''
    This function is for visualize knn performance
    :param filename: the dataset
    :param k: k for kNN
    :return:
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    def plot(Data, xx, yy, lambdaVs, levels):
        from matplotlib.colors import ListedColormap
        plt.figure(figsize=(12, 7))
        plt.contourf(xx, yy, lambdaVs, levels=levels, vmin=0, vmax=1)
        plt.colorbar()
        # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        # cmap = ListedColormap(colors[:len(np.unique(lambdaVs))])
        # cs = plt.contour(xx, yy, lambdaVs, cmap=cmap, levels=[.5])
        # cs.collections[0].set_label('Majority Vote Decision Boundary')
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

    def GenerateGrid(FileData, GridNum):
        '''
        This function generate the test data
        :param FileData: used to generate the range
        :param GridNum: test data num
        :return: test dataset
        '''

        def GenerateArrange(FileData):
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
        x = np.linspace(minX - 0.2 * xRange, maxX + 0.2 * xRange, GridNum)
        y = np.linspace(minY - 0.2 * yRange, maxY + 0.2 * yRange, GridNum)
        xx, yy = np.meshgrid(x, y)
        return xx, yy
    k = k
    xx, yy = GenerateGrid(FileData, GridNum=251)
    TestData=[]
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            TestData.append((xx[i][j], yy[i][j], '0'))
    lambdaVs = DLRTClassifier(FileData, TestData, k)
    lambdaVs = np.array(lambdaVs).reshape(xx.shape)
    plot(FileData, xx, yy, lambdaVs, levels=200)

def DLRTRoc(TrainData, TestData, k):
    '''
    Generate pd and pfa to plot roc curve
    :param TrainData:
    :param TestData:
    :param k:
    :return: pds, pfas and auc
    '''
    import RocPlotlib
    lambdaVs = DLRTClassifier(TrainData, TestData, k)
    pdfs = []
    for i in range(len(lambdaVs)):
        pdfs.append([float(TestData[i][2]), lambdaVs[i]])
    thresholds = RocPlotlib.GenerateThreshold(pdfs, 'decision 3')
    roc = RocPlotlib.GenerateRoc(pdfs, thresholds)
    return roc

class Solution():

    def question6a(self):
        from bayerslib import gen2DNormaldataset
        ds0, _, _ = gen2DNormaldataset("positive", '0', 200)
        ds1, _, _ = gen2DNormaldataset("negative", '1', 200)
        DLRTVisualization(ds0+ds1, k=5)

    def question7a(self):
        FileData = ReadDataFromFile("dataSetHorseshoes.csv")
        DLRTVisualization(FileData, k=3)
        DLRTVisualization(FileData, k=9)
    def question7b(self):
        from kNNlib import kNNVisualization
        kNNVisualization('dataSetHorseshoes.csv', 6)
        kNNVisualization('dataSetHorseshoes.csv', 18)

    def question7d(self):
        from crossValidation import getRoc
        from kNNlib import kNNRoc
        Data = ReadDataFromFile("dataSetHorseshoes.csv")
        getRoc(Data, 10, DLRTRoc, 3)
        getRoc(Data, 10, kNNRoc, 6)

if __name__ == "__main__":
    solution = Solution()
    solution.question7d()

