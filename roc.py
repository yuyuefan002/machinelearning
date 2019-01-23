import csv
import math
import matplotlib.pyplot as plt
from operator import itemgetter


def ReadDataFromFile(filename):
    '''
    This function read data from csv file while contains truth and lambda value.
    then return those values in list

    :param filename: the csv file from which we can get the truth and lambda
    :return: the list holding each line
    '''
    with open(filename, newline='') as csvfile:
        ClassiferOutput = []
        SpamReader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in SpamReader:
            ClassiferOutput.append(row[0].split(','))
        ClassiferOutput.sort(key=itemgetter(1))
        return ClassiferOutput


def case1(thresholds, ClassiferOutput):
    for truth, lambdaV in ClassiferOutput:
        thresholds.append(float(lambdaV))
    return thresholds


def case2(thresholds, ClassiferOutput):
    minlambda = math.inf
    for truth, value in ClassiferOutput:
        if(float(value)<minlambda):
            minlambda=float(value)

    _, maxlambda = max(ClassiferOutput, key=itemgetter(1))
    diff = (float(maxlambda) - float(minlambda)) / 99
    value = float(minlambda)
    i = 0
    while i < 99:
        thresholds.append(value)
        value += diff
        i += 1
    return thresholds


def case3(thresholds, ClassiferOutput):
    n = len(ClassiferOutput)
    step = int(n / 99)
    if step < 1:
        step = 1
    i = 0
    while i < n:
        truth, lambdaV = ClassiferOutput[i]
        thresholds.append(float(lambdaV))
        i += step
    return thresholds


def case4(thresholds, ClassiferOutput):
    for truth, lambdaV in ClassiferOutput:
        if float(truth) == 0:
            thresholds.append(float(lambdaV))
    return thresholds


def case5(thresholds, ClassiferOutput):
    h0 = []
    for truth, lambdaV in ClassiferOutput:
        if truth == '0':
            h0.append(float(lambdaV))
    i = 0
    n = len(h0)
    step = n / 100
    while i < n:
        thresholds.append(h0[int(i)])
        i += step
    return thresholds


def GenerateThreshold(ClassiferOutput, decision):
    '''
    This function generate the thresholds
    :param ClassiferOutput: including truth and lambda
    :return: the list of thresholds
    '''
    thresholds = []
    thresholds.append(-math.inf)
    if decision == 'decision 1':
        thresholds = case1(thresholds, ClassiferOutput)
    elif decision == 'decision 2':
        thresholds = case2(thresholds, ClassiferOutput)
    elif decision == 'decision 3':
        thresholds = case3(thresholds, ClassiferOutput)
    elif decision == 'decision 4':
        thresholds = case4(thresholds, ClassiferOutput)
    elif decision == 'decision 5':
        thresholds = case5(thresholds, ClassiferOutput)
    thresholds.append(math.inf)
    thresholds.sort()
    return thresholds


def CalP(ClassiferOutput, threshold):
    '''
    Calculate pfa and pd
    :param ClassiferOutput: including truth and lambda
    :param threshold: list of thresholds
    :return: pfa and pd
    '''
    FP = 0
    TN = 0
    TP = 0
    FN = 0
    for truth, lambdaV in ClassiferOutput:
        truth = float(truth)
        lambdaV = float(lambdaV)
        if truth == 0 and lambdaV >= threshold:
            FP += 1
        elif truth == 0 and lambdaV < threshold:
            TN += 1
        elif truth == 1 and lambdaV >= threshold:
            TP += 1
        elif truth == 1 and lambdaV < threshold:
            FN += 1
    pfa = FP/(FP+TN)
    pd = TP/(TP+FN)
    return pfa, pd


def GenerateRoc(ClassifierOutput, thresholds):
    '''
    Generate pd and pfa list to draw ROC
    :param ClassifierOutput: including truth and lambda
    :param thresholds:  list of threshold
    :return: ROC point list
    '''
    pds = []
    pfas = []
    for threshold in thresholds:
        pfa, pd = CalP(ClassifierOutput, threshold)
        pds.append(pd)
        pfas.append(pfa)
    return [pds, pfas]

def PlotRoc(ROCs):
    '''
    plot ROC
    :param roc: ROC points
    '''
    color = ['r', 'g', 'b', 'y', 'purple']
    label = ['label1', 'label2', 'label3', 'label4', 'label5',]
    plt.figure(figsize=(12, 7))
    plt.axis([0, 1, 0, 1])
    plt.title('ROC')

    plt.subplot(2,3,1)
    plt.plot([0, 1], [0, 1], '-.', linewidth=0.5, color='black')
    for i in range(0,5):
        plt.plot(ROCs[i][1], ROCs[i][0], color=color[i], label=label[i])
    plt.legend(loc='lower right')
    for i in range(0, 5):
        plt.subplot(2, 3, i+2)
        plt.plot([0, 1], [0, 1], '-.', linewidth=0.5, color='black')
        plt.plot(ROCs[i][1], ROCs[i][0], color=color[i], label=label[i])
        plt.legend(loc='lower right')
        if i == 2:
            plt.ylabel('Probability of Detection\n(P$_D$)')
            plt.xlabel('Probability of False Alarm\n(P$_F$$_A$)')
    plt.show()


def testcase(filename):
    ClassiferOutput = ReadDataFromFile(filename)
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 1')
    roc1 = GenerateRoc(ClassiferOutput, thresholds)
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 2')
    roc2 = GenerateRoc(ClassiferOutput, thresholds)
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 3')
    roc3 = GenerateRoc(ClassiferOutput, thresholds)
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 4')
    roc4 = GenerateRoc(ClassiferOutput, thresholds)
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 5')
    roc5 = GenerateRoc(ClassiferOutput, thresholds)
    ROCs = [roc1, roc2, roc3, roc4, roc5]
    PlotRoc(ROCs)


if __name__ == '__main__':
    testcase('smallData.csv')
