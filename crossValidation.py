import kNNlib
import csv



def ReadDataFromFile(filename):
    '''
    This function read data from csv file while contains truth and lambda value.
    then return those values in list

    :param filename: the csv file from which we can get the truth and lambda
    :return: the list holding each line
    '''
    with open(filename, newline='') as csvfile:
        FileData = []
        Fold1 = []
        Fold2 = []
        SpamReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in SpamReader:
            fold, classnum, feature1, feature2 = row[0].split(',')
            FileData.append((float(feature1), float(feature2), classnum))
            if fold == '1':
                Fold1.append((float(feature1), float(feature2), classnum))
            elif fold == '2':
                Fold2.append((float(feature1), float(feature2), classnum))
        return FileData, Fold1, Fold2

def aggregate(ROCs,numFolds):
    pds = []
    pfas = []
    pcdmax = 0
    for i in range(len(ROCs[0][0])):
        pd = 0
        pfa = 0
        for j in range(numFolds):
            pd += ROCs[j][0][i]
            pfa += ROCs[j][1][i]
        pds.append(pd / numFolds)
        pfas.append(pfa / numFolds)

    for i in range(numFolds):
        pcdmax += ROCs[i][2]
    agg_roc = [pds, pfas, pcdmax / numFolds]
    return agg_roc
def GetRocFromClassifier(ds0, ds1,numFolds, ROC, k=5):
    from numpy import remainder as rem
    from numpy import array
    from numpy.random import permutation as randperm
    ROCs = []
    keys0 = rem(array(range(0, len(ds0)))-1, numFolds)+1
    keys0 = keys0[randperm(len(keys0))]
    keys1 = rem(array(range(0, len(ds1)))-1, numFolds)+1
    keys1 = keys1[randperm(len(keys1))]
    for thisFold in range(1, numFolds+1):
        TrainData = []
        TestData = []
        for i in range(len(keys0)):
            if keys0[i] != thisFold:
                TrainData.append(ds0[i])
            elif keys0[i] == thisFold:
                TestData.append(ds0[i])
        for i in range(len(keys1)):
            if keys1[i] != thisFold:
                TrainData.append(ds1[i])
            elif keys1[i] == thisFold:
                TestData.append(ds1[i])
        roc = ROC(TrainData, TestData, k)
        ROCs.append(roc)
    return ROCs

def divideInto2Class(data):
    ds0 = []
    ds1 = []
    for datum in data:
        _, _, classnum = datum
        if classnum == '0':
            ds0.append(datum)
        elif classnum == '1':
            ds1.append(datum)
    return ds0, ds1

def plotDistribute(data, c=0):
    import matplotlib.pyplot as plt
    c1f1=[]
    c1f2=[]
    c2f1=[]
    c2f2=[]
    for datum in data:
        feature1,feature2,classnum=datum
        if classnum =='0':
            c1f1.append(feature1)
            c1f2.append(feature2)
        elif classnum == '1':
            c2f1.append(feature1)
            c2f2.append(feature2)
    color=['red','blue','yellow','green']
    label=['fold1 train class1','fold1 train class2','fold1 test class1','fold1 test class2',]
    plt.scatter(c1f1,c1f2,color=color[c],label=label[c])
    plt.scatter(c2f1, c2f2, color=color[c+1],label=label[c+1])
    plt.legend(loc='lower right')


def getRoc(Data, numFold, ROC, k):
    import RocPlotlib
    ds0, ds1 = divideInto2Class(Data)
    ROCs = GetRocFromClassifier(ds0, ds1, numFold, ROC, k=k)
    agg_roc = aggregate(ROCs, numFold)
    maxPcd = RocPlotlib.CalMaxPcd(agg_roc, 0.5, 0.5)
    ROCs = [agg_roc]
    label = ['agg_roc']
    RocPlotlib.plot(ROCs, label, maxPcds=[maxPcd])


class Solution:
    def question1a(self):
        import RocPlotlib
        TrainData, _, _ = ReadDataFromFile('dataSetCrossValWithKeys.csv')
        TestData = TrainData
        roc1 = kNNlib.kNNRoc(TrainData, TestData, k=5)
        ROCs = [roc1]
        label = ['k=5',]
        maxPcd=RocPlotlib.CalMaxPcd(roc1, 0.5, 0.5)
        RocPlotlib.plot(ROCs, label, maxPcds=[maxPcd])

    def question1b(self):
        import RocPlotlib
        _, Fold1, Fold2 =ReadDataFromFile('dataSetCrossValWithKeys.csv')
        TrainData = Fold2
        TestData = Fold1
        plotDistribute(TrainData)
        plotDistribute(TestData,c=2)
        roc1 = kNNlib.kNNRoc(TrainData, TestData, k=5)
        TrainData = Fold1
        TestData = Fold2
        roc2 = kNNlib.kNNRoc(TrainData, TestData, k=5)
        pds = []
        pfas = []
        for i in range(len(roc1[0])):
            pd = roc1[0][i]
            pfa = roc1[1][i]
            pd += roc2[0][i]
            pfa += roc2[1][i]
            pds.append(pd/2)
            pfas.append(pfa/2)
        agg_roc = [pds, pfas, (roc1[2]+roc2[2])/2]
        maxPcd=RocPlotlib.CalMaxPcd(agg_roc, 0.5, 0.5)
        ROCs = [agg_roc]
        label = ['agg_roc']
        RocPlotlib.plot(ROCs, label, maxPcds=[maxPcd])

    def question1e(self):
        import RocPlotlib
        from kNNlib import kNNRoc
        Data, _, _, = ReadDataFromFile('dataSetCrossValWithKeys.csv')
        ds0, ds1 = divideInto2Class(Data)
        ROCs = GetRocFromClassifier(ds0, ds1, 2, kNNRoc)
        agg_roc = aggregate(ROCs, 2)
        maxPcd=RocPlotlib.CalMaxPcd(agg_roc, 0.5, 0.5)
        ROCs = [agg_roc]
        label = ['agg_roc']
        RocPlotlib.plot(ROCs, label, maxPcds=[maxPcd])

    def question2a(self):

        def plot(data):
            import matplotlib.pyplot as plt
            x, minPes = data[0]
            print(data)
            plt.axis([0.0, 400.0, 0.0, 1.0])
            plt.plot(x, minPes, color='red',label="cross validation")
            x, minPes = data[1]
            plt.plot(x, minPes, color='blue', label="test on training data")
            x, minPes = data[2]
            plt.plot(x, minPes, color='green', label="test on testing data")
            plt.legend(loc='lower right')
            plt.show()
        import kNNlib
        from kNNlib import kNNRoc
        import RocPlotlib
        data = []
        minPes = []
        x = []
        Data = kNNlib.ReadDataFromFile('dataSetHorseshoes.csv')
        for k in range(1, 360):
            ds0, ds1 = divideInto2Class(Data)
            ROCs = GetRocFromClassifier(ds0, ds1, numFolds=10, ROC=kNNRoc, k=k)
            agg_roc = aggregate(ROCs, numFolds=10)
            maxPcd = RocPlotlib.CalMaxPcd(agg_roc, 0.5, 0.5)
            minPes.append(1 - maxPcd[0])
            x.append(400/k)
        data.append([x, minPes])
        TrainData = kNNlib.ReadDataFromFile('dataSetHorseshoes.csv')
        TestData = TrainData
        minPes = []
        x = []

        for k in range(1, 400):
            roc = kNNRoc(TrainData, TestData, k)
            maxPcd = RocPlotlib.CalMaxPcd(roc, 0.5, 0.5)
            minPes.append(1 - maxPcd[0])
            x.append(400/k)

        data.append([x, minPes])

        TrainData = kNNlib.ReadDataFromFile('dataSetHorseshoes.csv')
        TestData = kNNlib.ReadDataFromFile('dataSetHorseshoesTest.csv')
        minPes = []
        x = []
        for k in range(1, 400):
            roc = kNNRoc(TrainData, TestData, k)
            maxPcd = RocPlotlib.CalMaxPcd(roc, 0.5, 0.5)
            minPes.append(1 - maxPcd[0])
            x.append(400/k)
        data.append([x, minPes])

        plot(data)
if __name__=='__main__':
     import RocPlotlib
     solution = Solution()
     #solution.question1b()
     solution.question1e()
     #solution.question2a()



