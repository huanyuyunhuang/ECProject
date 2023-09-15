import numpy as np
import matplotlib.pyplot as plt
#不具有通用性
def fun(x):
    # return np.multiply(x,x-20) - 100 * np.sin(x) - 200 * np.cos(np.abs(x))
    return np.multiply(x,x-20) + 100
# 生成x轴上的点
x = np.linspace(-50, 50, 1000)

# 计算函数值
y = fun(x)

# 绘制曲线
plt.plot(x, y)

# 添加标题和轴标签
plt.title('Function Curve')
plt.xlabel('x')
plt.ylabel('y')


#定义粒子--V、pbest、bounds、position
class Particle:
    def __init__(self,bounds):
        self.velocity = np.zeros(1)
        self.position = np.random.uniform(bounds[0],bounds[1],size=1)
        self.pbest_pos = self.position.copy()
        self.pbest_fitness = None

    def fitness(self):
        fitness = fun(self.position[0])#替换成优化函数
        self.pbest_fitness = fitness
        return fitness
#定义粒子群及更新操作
class PSO:
    def __init__(self, num_particle, bounds, max_iterations):
        self.num_particle = num_particle
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.gbest_pos = None
        self.particles = []
        self.best_iteration = 0

    def init_particle(self):
        for _ in range(self.num_particle):
            particle = Particle(self.bounds)
            self.particles.append(particle)

    def update_particle(self, particle, w, c1, c2):
        rand1 = np.random.random()
        rand2 = np.random.random()

        particle.velocity = w * particle.velocity + c1 * rand1 * (particle.pbest_pos - particle.position) + c2 * rand2 * (self.gbest_pos - particle.position)
        particle.position = particle.position + particle.velocity

        #边界处理
        particle.position = np.clip(particle.position, bounds[0], bounds[1])

    def run(self):
        fitness_line = []
        self.init_particle()
        self.gbest_pos = self.particles[0].position.copy()#初始化gbest_pos

        for i in range(self.max_iterations):
            print(self.gbest_pos)
            # 生成x轴上的点
            x = np.linspace(-50, 50, 1000)
            # 计算函数值
            y = fun(x)
            # 绘制曲线
            plt.plot(x, y)
            for particle in self.particles:
                plt.plot(particle.position, fun(particle.position), 'go')
                plt.draw()
                fitness = particle.fitness()
                #更新局部
                if fitness < particle.pbest_fitness:
                    particle.pbest_pos=particle.position.copy()
                    particle.pbest_fitness=particle.fitness.copy()
                    fitness_line.append(particle.pbest_fitness)
                # 更新全局
                if fitness < fun(self.gbest_pos[0]):
                    self.gbest_pos = particle.position.copy()
                    self.best_iteration = i
                    fitness_line.append(fun(self.gbest_pos[0]))
                plt.plot(self.gbest_pos, fun(self.gbest_pos), 'ro')
                self.update_particle(particle, 0.5, 1.49, 1.49)
            plt.pause(0.01)
            plt.cla()

        return self.gbest_pos, self.best_iteration, fitness_line


# 运行PSO算法
num_particles = 10
bounds = [-50, 50]
max_iterations = 100

pso = PSO(num_particles, bounds, max_iterations)
best_position, best_iteration, fitness_line = pso.run()
# 显示图形
plt.close()
plt.show()

plt.figure()
plt.plot(fitness_line, label=f'fitness')
plt.legend()
plt.show()
# 输出最优解及其适应度
best_fitness = fun(best_position[0])
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
print("Best iteration:", best_iteration)




