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
def GetRocFromCrossValidation(ds0, ds1,numFolds):
    import RocPlotlib
    from numpy import remainder as rem
    from numpy import array
    from numpy.random import permutation as randperm
    ROCs=[]
    label=[]
    keys0 = rem(array(range(0, len(ds0)))-1, numFolds)+1
    keys0 = keys0[randperm(len(keys0))]
    keys1 = rem(array(range(0, len(ds1)))-1, numFolds)+1
    keys1 = keys0[randperm(len(keys0))]
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
        roc = kNNlib.kNNRoc(TrainData, TestData, k=5)
        ROCs.append(roc)
        #label.append('Fold {}'.format(thisFold))
    #RocPlotlib.plot(ROCs, label)
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


class Solution:
    def question1a(self):
        import RocPlotlib
        TrainData, _, _ = ReadDataFromFile('dataSetCrossValWithKeys.csv')
        TestData = TrainData
        roc1 = kNNlib.kNNRoc(TrainData, TestData, k=5)
        ROCs = [roc1]
        label = ['k=5',]
        RocPlotlib.plot(ROCs, label)

    def question1b(self):
        import RocPlotlib
        _, Fold1, Fold2 =ReadDataFromFile('dataSetCrossValWithKeys.csv')
        TrainData = Fold2
        TestData = Fold1
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
        ROCs = [agg_roc]
        label = ['agg_roc']
        RocPlotlib.plot(ROCs, label)

    def question1e(self):
        import RocPlotlib
        Data, _, _, = ReadDataFromFile('dataSetCrossValWithKeys.csv')
        ds0, ds1 = divideInto2Class(Data)
        ROCs = GetRocFromCrossValidation(ds0, ds1, 2)
        agg_roc = aggregate(ROCs, 2)
        ROCs = [agg_roc]
        label = ['agg_roc']
        RocPlotlib.plot(ROCs, label)
if __name__=='__main__':
     solution = Solution()
     solution.question1e()

