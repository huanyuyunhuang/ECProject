import numpy as np
from scipy.stats import norm
# 使用了SciPy库中的norm模块来建模高斯分布。
# 首先，使用norm.fit()函数拟合数据，得到估计的均值和标准差。
# 然后，使用norm.rvs()函数从估计的高斯分布中生成新的样本。
np.random.seed(123)  # 设置随机数种子
# univariate EDAs
# 准备数据集
initial_samples = [1.2, 2.5, 3.7, 4.1, 5.6]#实际问题中的问题解向量
# 估计概率分布模型
mean, std = norm.fit(initial_samples)
# 生成新的样本
new_samples = norm.rvs(loc=mean, scale=std, size=10)
print("Estimated mean:", mean)
print("Estimated standard deviation:", std)
print("Generated samples:", new_samples)

# incremental EDAs
# num_iterations = 10  # 迭代次数
# for i in range(num_iterations):
#     new_sample = norm.rvs(loc=mean, scale=std, size=1)  # 生成新的样本（根据具体问题和算法要求）
#     updated_samples = np.concatenate((initial_samples, new_sample))  # 更新样本集
#     print("updated_samples:", updated_samples)
#     updated_mean, updated_std = norm.fit(updated_samples)  # 更新均值和标准差
#     mean, std = updated_mean, updated_std
# print("updated mean:", mean)
# print("updated standard deviation:", std)



