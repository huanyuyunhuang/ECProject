import numpy as np
import matplotlib.pyplot as plt

def schaffer_f6(x1, x2):
    return 0.5 + ((np.sin(x1**2 - x2**2)**2 - 0.5) / (1 + 0.001 * (x1**2 + x2**2))**2)

x1 = np.linspace(-100, 100, 100)
x2 = np.linspace(-100, 100, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = schaffer_f6(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.set_title("Schaffer's f6 Function")

plt.show()