import numpy as np
import matplotlib.pyplot as plt

# 定义复杂一元多项函数
def f(x):
    return np.sin(x) + 0.2 * np.cos(3*x) + 0.1 * x**3 - 2 * np.sin(0.5*x)**2

# 定义定义域
x = np.linspace(-10, 10, 500)

# 计算函数值
y = f(x)

# 绘制函数图像
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Complex Univariate Polynomial Function')

# 找到最小值
min_x = x[np.argmin(y)]
min_y = np.min(y)

# 找到波峰和波谷
peaks, _ = scipy.signal.find_peaks(-y)
valleys, _ = scipy.signal.find_peaks(y)

# 确保至少有三个波峰和波谷
if len(peaks) < 3 or len(valleys) < 3:
    print("函数的波峰和波谷数量不足三个。请尝试调整函数或定义域范围。")
    exit()

# 标记最小值、波峰和波谷
plt.scatter(min_x, min_y, color='red', label='Minimum')
plt.scatter(x[peaks], y[peaks], color='green', label='Peaks')
plt.scatter(x[valleys], y[valleys], color='blue', label='Valleys')

# 显示图像
plt.legend()
plt.show()