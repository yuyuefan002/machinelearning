import random
import matplotlib.pyplot as plt


def GetFlipResult(D):
    result = []
    for i in range(D):
        dice = random.randint(1, 100)
        if dice <= 50:
            result.append(0)
        else:
            result.append(1)
    return result


def GetResult(P, Q, D):
    n = P + Q
    results = []
    for i in range(n):
        results.append(GetFlipResult(D))
    return results

def PQisDiff(P, Q, results):
    for i in range(0, P):
        for j in range(0, Q):
            if results[i] == results[P+j]:
                return False
    return True
    # for i in range(1, P):
    #     if results[i] != results[i-1]:
    #         return False
    # for j in range(1, Q):
    #     if results[P+j] != results[P+j-1]:
    #         return False
    #
    # for i in range(len(results[0])):
    #     if results[0][i] == results[P][i]:
    #         return False
    # return True

def MonteCarloSimulator(P, Q, D, times):
    different_count = 0
    table = []
    for i in range(times):
        results = GetResult(P, Q, D)
        veriResults = PQisDiff(P, Q, results)
        table.append([results, P, Q, veriResults])
        if veriResults == True:
            different_count += 1
    return different_count, table


P = 5
D = 5
x = []
y = []
for Q in range(1,50):
    total_times = 1000
    different_count, table = MonteCarloSimulator(P=P, Q=Q, D=D, times=total_times)
    x.append(Q)
    y.append(float(different_count)/total_times)
plt.plot(x, y)
plt.title('Probability when P={}, D={}'.format(P, D))
plt.xlabel('Q')
plt.ylabel('Probability')
plt.scatter(x[0], y[0], color="yellow", label='Q=1,Pb={}'.format(y[0]))
plt.scatter(x[1], y[1], color="red", label='Q=2,Pb={}'.format(y[1]))
plt.scatter(x[4], y[4], color="red", label='Q=5,Pb={}'.format(y[4]))
plt.legend(loc='upper right')
plt.grid(linestyle='-.')
plt.show()