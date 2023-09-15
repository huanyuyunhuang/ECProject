import numpy as np
import matplotlib.pyplot as plt

num_variables = 50 #优化参数数量
population_size = 100 #问题规模
probabilities_vector =np.random.rand(num_variables) #初始化概率向量
population = np.zeros((population_size, num_variables))
print(population)
max_iterations = 100
learning_rate = 0.01
finish_iteration = 0
finish_count = 0
draw_probabilities = []
draw_probabilities.append(probabilities_vector)

for iteration in range(max_iterations):
    fitness = np.sum(population, axis=1)
    if np.sum(fitness) == population_size * num_variables:
        finish_count += 1
    if finish_count > 5:
        break
    sorted_indices = np.argsort(-fitness)
    best_population = population[sorted_indices[0]]  # 获取排序后的第一个值，即最高项的值
    best_fitness = fitness[sorted_indices[0]]
    for i in range(num_variables):
        if best_population[i] == 1:
            probabilities_vector = (1 - learning_rate) * probabilities_vector + learning_rate * 1
        else:
            probabilities_vector = (1 - learning_rate) * probabilities_vector + learning_rate * 0
    draw_probabilities.append(probabilities_vector)
    new_population = np.zeros((population_size // 2, num_variables))
    for j in range(num_variables):
        new_population[:,j] = np.random.choice([0, 1], size=(population_size // 2, 1), p=[1-probabilities_vector[j], probabilities_vector[j]]).reshape((population_size // 2,))  # 构建初始种群
    last_population = population[np.argsort(fitness)][-population_size // 2:]
    population = np.concatenate((last_population, new_population))
    finish_iteration = iteration
    print("running")

print("population ",population)
print("probabilities_vector ",probabilities_vector)
print("finish_iteration ",finish_iteration)
draw_probabilities = np.transpose(np.array(draw_probabilities))
num_rows, num_cols = draw_probabilities.shape
plt.figure()
for i in range(num_rows):
    plt.plot(draw_probabilities[i])

plt.legend()
plt.show()