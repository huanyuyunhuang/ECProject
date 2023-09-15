import numpy as np
import matplotlib.pyplot as plt
#核心：从当前解中获得概率分布，再由此分布生成解
#onemax问题

class EDA:
    def __init__(self, onemax, max_iterations):
        self.onemax = onemax
        self.raw = self.onemax.shape[0]
        self.col = self.onemax.shape[1]
        self.vector_len = self.col
        self.max_iterations = max_iterations

    def run(self):
        draw_probabilistic_vector = []
        for i in range(self.max_iterations):
            print(self.onemax)
            print(np.sum(self.onemax))
            if np.sum(self.onemax) == self.raw * self.col:
                return self.onemax, result_iteration, draw_probabilistic_vector
            result_iteration = i
            fitness = np.sum(self.onemax, axis=1)
            #print(fitness)
            #fitness降序排序
            sorted_index = np.argsort(fitness)[::-1]
            better_vector=[]
            #取一半最优
            for i in range(int(len(sorted_index)/2)):
                better_vector.append(self.onemax[sorted_index[i]])
            better_vector = np.array(better_vector)
            #print(better_vector)
            #统计列的1数量,获得概率分布模型
            probabilistic_vector = np.sum(better_vector, axis=0) / better_vector.shape[0]
            draw_probabilistic_vector.append(probabilistic_vector)
            #print(probabilistic_vector)
            #生成新解,替代原onemax
            temp_onemax = []
            for i in range(self.raw):
                temp_vector = []
                for j in range(len(probabilistic_vector)):
                    temp_vector.append(np.random.choice([0, 1], p=[1 - probabilistic_vector[j], probabilistic_vector[j]]))
                temp_onemax.append(temp_vector)
            self.onemax = np.array(temp_onemax)
        return self.onemax, result_iteration, draw_probabilistic_vector



onemax = np.array([[1,0,0,1,1],
                   [0,1,0,0,0],
                   [1,0,0,1,0],
                   [0,1,1,0,0],
                   [0,0,1,0,1],
                   [0,1,0,0,1]])
#随机生成100*50的数组
# onemax = np.random.choice([0, 1], size=(1000,500))
max_iterations = 200
eda=EDA(onemax, max_iterations)
solution, result_iteration, draw_probabilistic_vector = eda.run()
print("onemax的解:", solution)
print("迭代次数:", result_iteration)

draw_probabilistic_vector = np.array(draw_probabilistic_vector)
draw_probabilistic_vector = np.transpose(draw_probabilistic_vector)
#print(draw_probabilistic_vector)
num_rows, num_cols = draw_probabilistic_vector.shape
plt.figure()
for i in range(num_rows):
    # plt.plot(draw_probabilistic_vector[i], label=f'Curve {i+1}')
    plt.plot(draw_probabilistic_vector[i])

plt.legend()
plt.show()