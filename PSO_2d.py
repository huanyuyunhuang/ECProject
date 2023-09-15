import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1554)
#不具有通用性
def fun(x1, x2):
    # return -200 * np.exp(-0.02 * np.sqrt(x1 * x1 + x2 * x2))
    return 2.15+x1*np.sin(4*np.pi*x1)+x2*np.sin(20*np.pi*x2)

#定义粒子--V、pbest、bounds、position
class Particle:
    def __init__(self, bounds_1, bounds_2):
        self.velocity = np.zeros(2)
        pos1 = np.random.uniform(bounds_1[0], bounds_1[1], size=1)
        pos2 = np.random.uniform(bounds_2[0], bounds_2[1], size=1)
        self.position = np.hstack((pos1, pos2))
        self.pbest_pos = self.position.copy()
        self.pbest_fitness = None

    def fitness(self):
        fitness = fun(self.position[0], self.position[1])#替换成优化函数
        self.pbest_fitness = fitness
        return fitness
#定义粒子群及更新操作
class PSO:
    def __init__(self, num_particle, bounds_1, bounds_2, max_iterations):
        self.num_particle = num_particle
        self.bounds_1 = bounds_1
        self.bounds_2 = bounds_2
        self.max_iterations = max_iterations
        self.gbest_pos = None
        self.particles = []
        self.best_iteration = 0

    def init_particle(self):
        for _ in range(self.num_particle):
            particle = Particle(self.bounds_1, self.bounds_2)
            self.particles.append(particle)

    def update_particle(self, particle, w, c1, c2):
        rand1 = np.random.random()
        rand2 = np.random.random()

        particle.velocity = w * particle.velocity + c1 * rand1 * (particle.pbest_pos - particle.position) + c2 * rand2 * (self.gbest_pos - particle.position)
        particle.position = particle.position + particle.velocity
        print(particle.position)
        #边界处理
        particle.position[0] = np.clip(particle.position[0], self.bounds_1[0], self.bounds_1[1])
        particle.position[1] = np.clip(particle.position[1], self.bounds_2[0], self.bounds_2[1])

    def run(self):
        fitness_line = []
        self.init_particle()
        self.gbest_pos = self.particles[0].position.copy()#初始化gbest_pos

        for i in range(self.max_iterations):
            print(self.gbest_pos)
            for particle in self.particles:
                fitness = particle.fitness()
                #更新局部
                if fitness < particle.pbest_fitness:
                    particle.pbest_pos=particle.position.copy()
                    particle.pbest_fitness=particle.fitness.copy()
                    fitness_line.append(particle.pbest_fitness)
                # 更新全局
                if fitness < fun(self.gbest_pos[0], self.gbest_pos[1]):
                    self.gbest_pos = particle.position.copy()
                    self.best_iteration = i
                    fitness_line.append(fun(self.gbest_pos[0], self.gbest_pos[1]))
                self.update_particle(particle, 0.5, 1.49, 1.49)

        return self.gbest_pos, self.best_iteration, fitness_line


# 运行PSO算法
num_particles = 20
bounds_1 = [-3, 12.1]
bounds_2 = [4.1, 5.8]
max_iterations = 3000

pso = PSO(num_particles, bounds_1, bounds_2, max_iterations)
best_position, best_iteration, fitness_line = pso.run()
best_fitness = fun(best_position[0], best_position[1])
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
print("Best iteration:", best_iteration)
plt.figure()
plt.plot(fitness_line, label=f'fitness')
plt.legend()
plt.show()





