import random

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1000)
#min问题,如果是max问题，函数加“-”
def fun(x1, x2):
    # return -200 * np.exp(-0.02 * np.sqrt(x1 * x1 + x2 * x2))
    return 2.15+x1*np.sin(4*np.pi*x1)+x2*np.sin(20*np.pi*x2)

population_size = 20
num_v = 2
bounds_1 = [-3,12.1]
bounds_2 = [4.1,5.8]
population_1 = np.random.uniform(bounds_1[0], bounds_1[1], size=(population_size, 1))
population_2 = np.random.uniform(bounds_2[0], bounds_2[1], size=(population_size, 1))
population = np.hstack((population_1, population_2))
print(population)
max_iteration = 1000
temp_dv = np.zeros((population_size, num_v))
exK = population_size // 2
temp_exK_dv = np.zeros((exK, num_v))
alpha = 0.1
fitness = np.zeros((population_size, 1))
draw = []
best_postion = []
best_fitness = np.inf

for iteration in range(max_iteration):
    for i in range(population_size):
        fitness[i] = fun(population[i][0], population[i][1])
    #draw.append(fun(population[np.argmax(fitness)][0], population[np.argmax(fitness)][1]))
    draw.append(np.min(fitness))
    mean_x = np.mean(population, axis=0)
    for i in range(population_size):
        temp_dv[i] = population[i] - mean_x
    dv = np.sqrt(np.mean(pow(temp_dv, 2), axis=0))
    sortfitness = np.argsort(fitness, axis=0)
    X_worst = population[sortfitness[population_size-1][0]]
    X_best = population[sortfitness[0][0]]
    X_Sebest = population[sortfitness[1][0]]

    if np.min(fitness) < best_fitness:
        best_fitness = np.min(fitness)
        best_postion = X_best

    mean_x = (1-alpha) * mean_x + alpha * (X_best + X_Sebest - X_worst)
    mean_exK = 0
    for i in range(exK):
        mean_exK += population[sortfitness[i][0]]
    mean_exK /= exK
    for i in range(exK):
        temp_exK_dv[i] = population[sortfitness[i][0]] - mean_x
    exK_dv = np.sqrt(np.mean(pow(temp_exK_dv, 2), axis=0))
    dv = (1 - alpha) * dv + alpha * exK_dv
    print("mean:", mean_x)
    print("variances:", dv)
    gen_population_1 = np.random.normal(loc=mean_x[0], scale=dv[0], size=(exK, 1))
    gen_population_2 = np.random.normal(loc=mean_x[1], scale=dv[1], size=(exK, 1))
    #边界处理
    gen_population_1 = np.clip(gen_population_1, bounds_1[0], bounds_1[1])
    gen_population_2 = np.clip(gen_population_2, bounds_2[0], bounds_2[1])
    gen_population = np.hstack((gen_population_1, gen_population_2))

    new_population = []
    for i in range(exK):
        new_population.append(gen_population[i])
        new_population.append(population[sortfitness[i][0]])
    population = new_population


print("solution_x:", best_postion)
print("solution_y:", best_fitness)
plt.figure()
plt.plot(draw)
plt.legend()
plt.show()
