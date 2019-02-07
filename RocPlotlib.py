import csv
import math
import matplotlib.pyplot as plt
from operator import itemgetter
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
        SpamReader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in SpamReader:
            truth, lambdaV = row[0].split(',')
            FileData.append([float(truth), float(lambdaV)])
        FileData.sort(key=itemgetter(1))
        return FileData

def plotDistribute(ClassiferOutput):
    truths = []
    lambdaVs = []
    for truth, lambdaV in ClassiferOutput:
        truths.append(float(truth))
        lambdaVs.append(float(lambdaV))
    plt.scatter(lambdaVs, truths)
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
        if truth == 0:
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
    pfas.reverse()
    pds.reverse()
    auc = np.trapz(pds, pfas)
    return [pds, pfas, auc]

def PlotRoc(ROCs):
    '''
    plot ROC for question 9
    :param roc: ROC points
    '''
    color = ['r', 'g', 'b', 'y', 'purple']
    label = ['app1', 'app2', 'app3', 'app4', 'app5',]
    plt.figure(figsize=(12, 7))
    plt.axis([-0.01, 1.01, -0.01, 1.01])
    plt.title('ROC')
    plt.plot([0, 1], [0, 1], '-.', linewidth=0.5, color='black')
    for i in range(len(ROCs)):
        plt.plot(ROCs[i][1], ROCs[i][0], color=color[i], label=label[i], linewidth=len(ROCs)/(i+1))
    plt.legend(loc='lower right')
    plt.ylabel('Probability of Detection\n(P$_D$)')
    plt.xlabel('Probability of False Alarm\n(P$_F$$_A$)')
    plt.show()
    plt.subplot(2, 3, 1)
    plt.plot([0, 1], [0, 1], '-.', linewidth=0.5, color='black')
    for i in range(0, len(ROCs)):
        plt.plot(ROCs[i][1], ROCs[i][0], color=color[i], label=label[i], linewidth=5/(i+1))
    plt.legend(loc='lower right')

    for i in range(0, 5):
        plt.subplot(2, 3, i+2)
        plt.plot([0, 1], [0, 1], '-.', linewidth=0.5, color='black')
        plt.plot(ROCs[i][1], ROCs[i][0], color=color[i], label=label[i])
        plt.text(0.8, 0.2, 'auc:{:0.2f}'.format(ROCs[i][2]))
        plt.legend(loc='lower right')
        if i == 2:
            
            plt.ylabel('Probability of Detection\n(P$_D$)')
            plt.xlabel('Probability of False Alarm\n(P$_F$$_A$)')

    # plt.ylabel('Probability of Detection\n(P$_D$)')
    # plt.xlabel('Probability of False Alarm\n(P$_F$$_A$)')
    plt.show()


def PlotSingleRoc(roc, maxPcd1, maxPcd2, maxPcd3):
    '''
    Plot the picture for question 12
    :param roc:
    :param maxPcd1:
    :param maxPcd2:
    :param maxPcd3:
    :return:
    '''
    color = ['r', 'g', 'b', 'y', 'purple']
    label = ['app1', 'app2', 'app3', 'app4', 'app5', ]
    plt.figure(figsize=(12, 7))
    plt.axis([-0.005, 1.005, -0.005, 1.005])
    plt.title('ROC')

    plt.plot([0, 1], [0, 1], '-.', linewidth=0.5, color='black')
    plt.scatter(x=maxPcd1[2], y=maxPcd1[1], color='red', label='p(H0)=p(H1)')
    plt.annotate(f'{maxPcd1[0]}', (maxPcd1[2]+0.02, maxPcd1[1]+0.02))
    plt.scatter(x=maxPcd2[2], y=maxPcd2[1], color='purple', label='2p(H0)=p(H1)')
    plt.annotate(f'{maxPcd2[0]}', (maxPcd2[2]+0.02, maxPcd2[1]+0.02))
    plt.scatter(x=maxPcd3[2], y=maxPcd3[1], color='yellow', label='p(H0)=2p(H1)')
    plt.annotate('{:.4f}'.format(maxPcd3[0]), (maxPcd3[2]+0.02, maxPcd3[1]-0.02))
    plt.plot(roc[1], roc[0], color=color[2], label='roc')
    plt.text(0.9, 0.2, 'auc:{:0.2f}'.format(roc[2]))
    plt.legend(loc='lower right')
    plt.ylabel('Probability of Detection\n(P$_D$)')
    plt.xlabel('Probability of False Alarm\n(P$_F$$_A$)')
    plt.show()


def CalculateOverlapPoint(points):
    '''
    This function calculate how many points are overlapped in the roc
    :param points: points coordinate
    :return: number of overlapped points
    '''
    count = 0
    for i in range(1, len(points[0])):
        if points[0][i] == points[0][i-1] and points[1][i] == points[1][i-1]:
            count += 1
    return count


def ThresholdComparison(filename):
    '''
    Test function for question 9
    :param filename:
    :return:
    '''
    ClassiferOutput = ReadDataFromFile(filename)
    plotDistribute(ClassiferOutput)
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 1')
    roc1 = GenerateRoc(ClassiferOutput, thresholds)
    print(CalculateOverlapPoint(roc1))
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 2')
    roc2 = GenerateRoc(ClassiferOutput, thresholds)
    print(CalculateOverlapPoint(roc2))
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 3')
    roc3 = GenerateRoc(ClassiferOutput, thresholds)
    print(CalculateOverlapPoint(roc3))
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 4')
    roc4 = GenerateRoc(ClassiferOutput, thresholds)
    print(CalculateOverlapPoint(roc4))
    thresholds = GenerateThreshold(ClassiferOutput, 'decision 5')
    roc5 = GenerateRoc(ClassiferOutput, thresholds)
    print(CalculateOverlapPoint(roc5))
    ROCs = [roc1, roc2, roc3, roc4, roc5]
    PlotRoc(ROCs)


def plot(ROCs, label, maxPcds=None, points=None, mean_point=None):
    '''
    plot ROC for question 9
    :param roc: ROC points
    '''

    color = ['purple', 'g', 'b', 'y', 'red']
    label = label
    plt.figure(figsize=(12, 7))
    plt.axis([-0.01, 1.01, -0.01, 1.01])
    plt.title('ROC')
    plt.plot([0, 1], [0, 1], '-.', linewidth=0.5, color='black')
    for i in range(len(ROCs)):
        plt.plot(ROCs[i][1], ROCs[i][0], color=color[i], label=label[i]+" auc={:.3f}".format(ROCs[i][2]), linewidth=len(ROCs) / (i + 1))
    plt.legend(loc='lower right')
    plt.ylabel('Probability of Detection\n(P$_D$)')
    plt.xlabel('Probability of False Alarm\n(P$_F$$_A$)')
    if maxPcds is not None:
        for maxPcd in maxPcds:
            plt.scatter(x=maxPcd[2], y=maxPcd[1], color='red', label=f'{maxPcd[0]}')
            plt.annotate('{:.3f}'.format(maxPcd[0]), (maxPcd[2] +0.01, maxPcd[1] -0.01))
    if points is not None:
        for point in points:
            plt.scatter(x=point[1], y=point[0], color='red')
    if mean_point is not None:
        plt.scatter(x=point[1], y=point[0], color='yellow')
    plt.show()

def PlotROC(filename, thresholdStrategy):
    ClassiferOutput = ReadDataFromFile(filename)
    thresholds = GenerateThreshold(ClassiferOutput, thresholdStrategy)
    roc = GenerateRoc(ClassiferOutput, thresholds)
    print(CalculateOverlapPoint(roc))
    ROCs = [roc]
    plot(ROCs)
def FormatTransform(data):
    '''
    Transform the format of csv data
    :param data:
    :return:
    '''
    pfas = []
    pds = []
    for datum in data:
        pfas.append(float(datum[0]))
        pds.append(float(datum[1]))
    auc = np.trapz(pds, pfas)
    return [pds, pfas, auc]


def CalculateMaxPcd(roc_data, ph0, ph1):
    '''
    This function calculate the max pcd based on different conditions
    :param roc_raw:
    :param ph0:
    :param ph1:
    :return:
    '''
    max = 0
    max_pfa = 0
    max_pd = 0
    for pfa, pd in roc_data:
        tmp = float(pd) * ph1 + (1 - float(pfa)) * ph0
        if tmp > max:
            max = tmp
            max_pfa = float(pfa)
            max_pd = float(pd)
    return [max, max_pd, max_pfa]

def CalMaxPcd(roc, ph0, ph1):
    max= 0
    max_pfa = 0
    max_pd = 0
    pfa = roc[1]
    pd = roc[0]

    for i in range(len(pfa)):
        tmp = pd[i]*ph1 + (1-pfa[i]) * ph0
        if tmp > max:
            max = tmp
            max_pfa = pfa[i]
            max_pd = pd[i]
    return [max, max_pd, max_pfa]


def auc_pcd_test(filename):
    '''
    Test function for question 12
    :param filename:
    :return:
    '''
    roc_raw = ReadDataFromFile(filename)
    maxPcd1 = CalculateMaxPcd(roc_raw, 0.5, 0.5)
    maxPcd2 = CalculateMaxPcd(roc_raw, 1/3, 2/3)
    maxPcd3 = CalculateMaxPcd(roc_raw, 2/3, 1/3)
    roc = FormatTransform(roc_raw)
    PlotSingleRoc(roc, maxPcd1, maxPcd2, maxPcd3)


if __name__ == '__main__':
    #ThresholdComparison('moderateData.csv')
    auc_pcd_test('rocData.csv')
