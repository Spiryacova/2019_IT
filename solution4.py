# -*- coding: utf-8 -*-
# Задание 4:
# Входные данные: X, shape=(N, M). N - число точек, M - число координат
# 1. Генератор точек Парето (P) фронта
# 2. Нарисовать X, P в 'полярных' координатах (M осей)
# Общие замечание: все по возможности векторизовать в numpy до 2D матриц

import numpy as np
import matplotlib.pyplot as plt



# Какие-то исходные данные
n = 5
m = 5
X = np.random.sample((n, m))

# Вычисления
def is_pareto(x, X):
    if (np.sum(x == X.shape[1]) > 1):
        return False
    return True

pareto = np.empty([0, X.shape[1]])
not_pareto = np.empty([0, X.shape[1]])
all_ones = np.empty([X.shape[1], 1])
all_ones.fill(True)

for i in range(X.shape[0]):
    A = X[i]<=X
    result = A.dot(all_ones)
    if is_pareto(result, X):
        pareto = np.vstack((pareto, X[i]))
    else:
        not_pareto = np.vstack((not_pareto, X[i]))
        
# График
fig = plt.figure(figsize=[15, 15])
axs = fig.add_subplot(212, projection="polar")
axs.set_title('Polar')
axs.yaxis.grid(False)
axs.set_yticks([])
plt.thetagrids(np.arange(0, 360, 360.0/X.shape[1]), labels=np.arange(0, X.shape[1], 1))

xP = np.arange(0, X.shape[1], 1)
xP = np.append(xP, 0)
xP = xP * 2 * 3.14 / X.shape[1]

for x in not_pareto:    
    yP = np.append(x, x[0])
    axs.plot(xP, yP, color="black")
for x in pareto:
    yP = np.append(x, x[0])
    axs.plot(xP, yP, color="red")
plt.show()