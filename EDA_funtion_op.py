import random

import numpy as np
import matplotlib.pyplot as plt

#-35-35
def fun(x):
    return -(x * (x -10))



population_size = 10
num_v = 1
bounds = [-10,10]
population = np.random.uniform(-35, 35, size=(population_size, num_v))
print(population)
max_iteration = 100
temp_dv = np.zeros((population_size, num_v))
exK = population_size // 2
temp_exK_dv = np.zeros((exK, num_v))
alpha = 0.1
fitness = np.zeros((population_size, 1))
draw = []

for iteration in range(max_iteration):
    for i in range(population_size):
        fitness[i] = fun(population[i])
    draw.append(fun(population[np.argmax(fitness)]))
    mean_x = np.mean(population)
    for i in range(population_size):
        temp_dv[i] = population[i] - mean_x
    dv = np.sqrt(np.mean(pow(temp_dv, 2)))
    sortfitness = np.argsort(-fitness, axis=0)
    X_worst = population[sortfitness[population_size-1][0]]
    X_best = population[sortfitness[0][0]]
    X_Sebest = population[sortfitness[1][0]]

    mean_x = (1-alpha) * mean_x + alpha * (X_best + X_Sebest - X_worst)
    mean_exK = 0
    for i in range(exK):
        mean_exK += population[sortfitness[i][0]]
    mean_exK /= exK
    for i in range(exK):
        temp_exK_dv[i] = population[sortfitness[i][0]] - mean_x
    exK_dv = np.sqrt(np.mean(pow(temp_exK_dv, 2)))
    dv = (1 - alpha) * dv + alpha * exK_dv
    print("mean:", mean_x)
    print("variances:", dv)
    gen_population = np.random.normal(loc=mean_x, scale=dv, size=(exK, 1))

    new_population = []
    for i in range(exK):
        new_population.append(gen_population[i])
        new_population.append(population[sortfitness[i][0]])
    population = new_population

for i in range(population_size):
    fitness[i] = fun(population[i])
result = np.argmax(fitness)
print("solution_x:", population[result])
print("solution_y:", fun(population[result]))
plt.figure()
plt.plot(draw)
plt.legend()
plt.show()



