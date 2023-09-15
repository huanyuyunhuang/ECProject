import numpy as np

# 定义问题
num_variables = 10  # 变量数量
population_size = 50  # 种群大小
learning_rate = 0.0001  # 学习率
mutation_rate = 0.01  # 变异率
max_iterations = 100  # 最大迭代次数

# 初始化种群
population = np.random.choice([0, 1], size=(population_size, num_variables))

# 迭代优化
for iteration in range(max_iterations):
    # 计算适应度
    fitness = np.sum(population, axis=1)  # 适应度函数示例：求解二进制向量中1的个数

    # 选择父代
    parents = population[np.argsort(fitness)][-population_size // 2:]

    # 更新参数向量
    parameter_vector = np.mean(parents, axis=0)
    print(parameter_vector)
    # 变异操作
    # 生成1的概率pi*(1-lr)+lr*i
    # mutation = np.zeros((population_size // 2, num_variables), dtype=int)
    # for j in range(num_variables):
    #     pj = parameter_vector[j] * (1 - learning_rate) + learning_rate * j
    #     mutation[:, j] = np.random.choice([0, 1], size=population_size // 2, p=[1 - pj, pj])
    # mutated_parents = mutation
    mutation = np.random.choice([0, 1], size=(population_size // 2, num_variables),
                                p=[1 - mutation_rate, mutation_rate])
    mutated_parents = np.logical_xor(parents, mutation).astype(int)# 逻辑异或

    # 生成下一代种群
    population = np.concatenate((parents, mutated_parents))

# 输出结果
best_solution = population[np.argmax(fitness)]
print("最优解：", best_solution)
print("最优解对应的适应度：", np.sum(best_solution))