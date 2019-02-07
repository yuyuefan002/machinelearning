import csv

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


def kNNVisualization(filename, k):
    '''
    This function is for visualize knn performance
    :param filename: the dataset
    :param k: k for kNN
    :return:
    '''

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
        x = np.linspace(minX - 0.2 * xRange, maxX + 0.2 * xRange, GridNum)
        y = np.linspace(minY - 0.2 * yRange, maxY + 0.2 * yRange, GridNum)
        xx, yy = np.meshgrid(x, y)
        return xx, yy
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


def kNNRoc(TrainData, TestData, k):
    '''
    Generate pd and pfa to plot roc curve
    :param TrainData:
    :param TestData:
    :param k:
    :return: pds, pfas and auc
    '''
    import RocPlotlib
    lambdaVs = runClassifier(TrainData, TestData, k)
    pdfs = []
    for i in range(len(lambdaVs)):
        pdfs.append([float(TestData[i][2]), lambdaVs[i]])
    thresholds = RocPlotlib.GenerateThreshold(pdfs, 'decision 3')
    roc = RocPlotlib.GenerateRoc(pdfs, thresholds)
    return roc
class Solution:
    def question3a(self):
        kNNVisualization('dataSetHorseshoes.csv', 1)
        kNNVisualization('dataSetHorseshoes.csv', 5)
        kNNVisualization('dataSetHorseshoes.csv', 31)
        kNNVisualization('dataSetHorseshoes.csv', 91)

    def question3c(self):
        FileData = ReadDataFromFile('dataSetHorseshoes.csv')
        kNNVisualization('dataSetHorseshoes.csv', int(len(FileData) / 2 - 1))
        kNNVisualization('dataSetHorseshoes.csv', int(len(FileData) - 1))

    def question4a(self):
        import RocPlotlib
        TrainData = ReadDataFromFile('dataSetHorseshoes.csv')
        TestData = TrainData
        roc1 = kNNRoc(TrainData, TestData, 1)
        maxPcd1 = RocPlotlib.CalMaxPcd(roc1, 0.5, 0.5)
        roc2 = kNNRoc(TrainData, TestData, 5)
        maxPcd2 = RocPlotlib.CalMaxPcd(roc2, 0.5, 0.5)
        roc3 = kNNRoc(TrainData, TestData, 31)
        maxPcd3 = RocPlotlib.CalMaxPcd(roc3, 0.5, 0.5)
        roc4 = kNNRoc(TrainData, TestData, 91)
        maxPcd4 = RocPlotlib.CalMaxPcd(roc4, 0.5, 0.5)
        ROCs = [roc1, roc2, roc3, roc4]
        label = ['k=1', 'k=5', 'k=31', 'k=91', 'app5', ]
        RocPlotlib.plot(ROCs, label, maxPcds=[maxPcd1, maxPcd2, maxPcd3, maxPcd4, ])

    def question4c(self):
        import RocPlotlib
        TrainData = ReadDataFromFile('dataSetHorseshoes.csv')
        TestData = ReadDataFromFile('dataSetHorseshoesTest.csv')
        roc1 = kNNRoc(TrainData, TestData, 1)
        maxPcd1 = RocPlotlib.CalMaxPcd(roc1, 0.5, 0.5)
        roc2 = kNNRoc(TrainData, TestData, 5)
        maxPcd2 = RocPlotlib.CalMaxPcd(roc2, 0.5, 0.5)
        roc3 = kNNRoc(TrainData, TestData, 31)
        maxPcd3 = RocPlotlib.CalMaxPcd(roc3, 0.5, 0.5)
        roc4 = kNNRoc(TrainData, TestData, 91)
        maxPcd4 = RocPlotlib.CalMaxPcd(roc4, 0.5, 0.5)
        ROCs = [roc1, roc2, roc3, roc4]
        label = ['k=1', 'k=5', 'k=31', 'k=91', 'app5', ]
        RocPlotlib.plot(ROCs, label, maxPcds=[maxPcd1, maxPcd2, maxPcd3, maxPcd4, ])

    def question4e(self):
        def plot(data):
            x, minPes = data[0]
            print(data)
            plt.axis([0.0, 400.0, 0.0, 1.0])
            plt.plot(x, minPes, color='red')
            x, minPes = data[1]
            plt.plot(x, minPes, color='blue')
            plt.show()

        import RocPlotlib
        data = []
        TrainData = ReadDataFromFile('dataSetHorseshoes.csv')

        TestData = TrainData
        minPes = []
        x = []
        for k in range(1, 400):
            roc = kNNRoc(TrainData, TestData, k)
            maxPcd = RocPlotlib.CalMaxPcd(roc, 0.5, 0.5)
            minPes.append(1 - maxPcd[0])
            x.append(400/k)

        data.append([x, minPes])

        # save = []
        # for k in range(len(x)):
        #     save.append([x[k], minPes[k]])
        # with open('pe1.csv', 'w') as writefile:
        #     writer = csv.writer(writefile)
        #     writer.writerows(save)
        TestData = ReadDataFromFile('dataSetHorseshoesTest.csv')
        minPes = []
        x = []
        for k in range(1, 400):
            roc = kNNRoc(TrainData, TestData, k)
            maxPcd = RocPlotlib.CalMaxPcd(roc, 0.5, 0.5)
            minPes.append(1 - maxPcd[0])
            x.append(400/k)
        # save = []
        # for k in range(len(x)):
        #     save.append([x[k], minPes[k]])
        # with open('pe2.csv', 'w') as writefile:
        #     writer = csv.writer(writefile)
        #     writer.writerows(save)
        data.append([x, minPes])
        plot(data)

    def question5a(self):
        def seperate(ClassiferOutput):
            h0 = []
            h1 = []
            for data in ClassiferOutput:
                if data[0] == 0:
                    h0.append(data[1])
                else:
                    h1.append(data[1])
            return h0, h1
        import RocPlotlib
        import seaborn as sns
        sns.set()
        ClassiferOutput = RocPlotlib.ReadDataFromFile('knn3DecisionStatistics.csv')
        h0, h1 = seperate(ClassiferOutput)
        sns.kdeplot(h0)
        thresholds = [0, 0.33333, 0.66667, 1, math.inf]
        roc = RocPlotlib.GenerateRoc(ClassiferOutput, thresholds)
        ROCs = [roc]
        RocPlotlib.plot(ROCs, ['roc'])

    def question5(self):
        def randomGeneratePd95():
            pdcount = 0
            for i in range(20):
                if random.uniform(0, 1) >= 0.25:
                    pdcount += 1
            pfacount = 0
            for i in range(70):
                if random.uniform(0, 1) >= 0.25:
                    pfacount += 1
            pdpb = pdcount / 20
            pfapb = pfacount / 70
            pdpb = 0.8 + 0.2 * pdpb
            pfapb = 0.3 + 0.7 * pfapb
            return pdpb, pfapb
        def mean(points):
            n = len(points)
            mean_pfa = 0
            mean_pd = 0
            for point in points:
                mean_pd += point[0]
                mean_pfa += point[1]
            mean_pd = mean_pd / n
            mean_pfa = mean_pfa / n
            return [mean_pd, mean_pfa]
        import RocPlotlib
        import random
        ClassiferOutput = RocPlotlib.ReadDataFromFile('knn3DecisionStatistics.csv')
        thresholds = [0, 0.33333, 0.66667, 1, math.inf]
        roc = RocPlotlib.GenerateRoc(ClassiferOutput, thresholds)
        ROCs = [roc]
        points = []
        for i in range(1000):
            pdpb, pfapb = randomGeneratePd95()
            points.append([pdpb, pfapb])
        mean_point = mean(points)
        print(mean_point)
        RocPlotlib.plot(ROCs, ['roc'], points=points,mean_point=mean_point)


if __name__ == '__main__':
    Solution = Solution()
    # question 3a
    #Solution.question3a()
    # question 3c
    #Solution.question3c()
    # question 4a
    #Solution.question4a()
    # question 4c
    #Solution.question4c()
    #Solution.question4e()
    Solution.question5()