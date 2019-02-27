
def BayesVisualization(ds0,f01,f02,ds1,f11,f12):
    '''
    This function is for visualize knn performance
    :param filename: the dataset
    :param k: k for kNN
    :return:
    '''
    import numpy as np

    def plot(Data, xx, yy, lambdaVs, levels):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        plt.figure(figsize=(12, 7))
        plt.contourf(xx, yy, lambdaVs, levels=levels)
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
    ds0_ = [f01, f02]
    ds1_ = [f11, f12]
    xx, yy = GenerateGrid(ds0+ds1, GridNum=251)
    TestData=[]
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            TestData.append((xx[i][j], yy[i][j], '0'))
    miu0, cov0 = trainBayes(ds0_)
    p0 = 0.5
    miu1, cov1 = trainBayes(ds1_)
    p1 = 0.5
    lambdaVs = bayesClassifier([[miu0, cov0, p0], [miu1, cov1, p1]], TestData)
    lambdaVs = np.array(lambdaVs).reshape(xx.shape)
    plot(ds0+ds1, xx, yy, lambdaVs, levels=200)

def bayesClassifier(TrainData, TestData):
    import numpy as np
    from numpy import log, transpose, matmul
    from numpy.linalg import inv, norm
    lambdas = []
    d0, d1 = TrainData
    miu0, cov0, p0 = d0
    miu1, cov1, p1 = d1
    for data in TestData:
        x = np.array([data[0], data[1]])
        lambdaV = (-(1/2)*matmul(matmul(transpose(x-miu1), inv(cov1)), (x-miu1))-(1/2)*log(norm(cov1))+log(p1))
        lambdaV -= (-(1/2)*matmul(matmul(transpose(x-miu0), inv(cov0)), (x-miu0))-(1/2)*log(norm(cov0)+log(p0)))
        lambdas.append(lambdaV)
    return lambdas
def trainBayes(ds):
    import numpy as np
    # mean
    miu = []
    for feature in ds:
        temp = np.mean(feature)
        miu.append(temp)
    miu = np.array(miu)
    # cov
    cov = np.cov(np.stack((ds[0], ds[1]), axis=0))
    return miu, cov

def gen2DNormaldataset(correlation="positive",classnum="0",size=200):
    from numpy.random import multivariate_normal
    data = []
    if correlation == "positive":
        cov = [[2, 2], [2, 3]]
        mean = [2, 2]
    elif correlation == "negative":
        cov = [[2, -2], [-2, 5]]
        mean = [-2, -2]
    elif correlation == "independent0":
        cov = [[2, 0], [0, 3]]
        mean = [2, 2]
    elif correlation == "independent1":
        cov = [[2, 0], [0, 5]]
        mean = [-2, -2]
    elif correlation == "dependent0":
        cov = [[2,3],[3,4]]
        mean = [2, 2]
    elif correlation == "dependent1":
        cov = [[2,3],[3,4]]
        mean = [-2, -2]
    elif correlation == "independent20":
        cov = [[2,0],[0,4]]
        mean = [2, 2]
    elif correlation == "independent21":
        cov = [[2,0],[0,4]]
        mean = [-2, -2]
    f1, f2 = multivariate_normal(mean, cov, size).T
    for i in range(len(f1)):
        data.append((f1[i], f2[i], classnum))
    return data, f1, f2

class Solution:
    def question3(self):
        import matplotlib.pyplot as plt
        ds0, f1, f2 = gen2DNormaldataset("positive", 0, 200)
        plt.scatter(x=f1, y=f2, color="blue", label='class 0')
        ds1, f1, f2 = gen2DNormaldataset("negative", "1", 200)
        plt.scatter(x=f1, y=f2, color="red", label='class 1')
        plt.legend(loc='lower right')
        plt.show()

    def question4a(self):
        ds0, f01, f02 = gen2DNormaldataset("positive", '0', 200)
        ds1, f11, f12 = gen2DNormaldataset("negative", '1', 200)
        BayesVisualization(ds0, f01, f02, ds1, f11, f12)

    def question4c(self):
        ds0, f01, f02 = gen2DNormaldataset("independent0", '0', 200)
        ds1, f11, f12 = gen2DNormaldataset("independent1", '1', 200)
        BayesVisualization(ds0, f01, f02, ds1, f11, f12)

    def question4e(self):
        ds0, f01, f02 = gen2DNormaldataset("dependent0", '0', 200)
        ds1, f11, f12 = gen2DNormaldataset("dependent1", '1', 200)
        BayesVisualization(ds0, f01, f02, ds1, f11, f12)
    def question4g(self):
        ds0, f01, f02 = gen2DNormaldataset("independent20", '0', 200)
        ds1, f11, f12 = gen2DNormaldataset("independent21", '1', 200)
        BayesVisualization(ds0, f01, f02, ds1, f11, f12)
if __name__ == "__main__":
    solution = Solution()
    solution.question4e()
