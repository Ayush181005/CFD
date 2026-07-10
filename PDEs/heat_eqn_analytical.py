import numpy as np
import matplotlib.pyplot as plt

# constants for this function
L = np.pi # length of the rod
k = 1 # thermal diffusivity
totalT = .5 # total time to simulate

n = 5 # number of terms to include from the series solution
nx = 20 # number of points to plot
nt = 50 # number of points to plot

x = np.linspace(0, L, nx)
t = np.linspace(0, totalT, nt)

# Initial Conditions
u = 100*np.ones(nx)

def calculate_y(xi, ti):
    global u, n, L, k
    u = np.zeros(nx)
    for i in range(1, n-1):
        u += (200/(i*np.pi))*(1-(-1)**i)*np.sin(i*np.pi*xi/L)*np.exp(-k*(i*np.pi/L)**2*ti)

# print(y)

# 2D Solution plot, comparing initial and final states
plt.plot(x, u)
calculate_y(x, totalT)
plt.plot(x, u)
plt.show()

# # plot surface
# X, T = np.meshgrid(x, t)
# U = np.zeros((nt, nx))

# for i in range(nt):
#     calculate_y(x, t[i])
#     U[i, :] = u
#     y = np.zeros(nx)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, U)
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel("u(x, t)")
# plt.title(r"Heat (1D Diffusion) Equation PDE: $k\frac{\partial^2u}{\partial x^2} = \frac{\partial u}{\partial t}$")
# plt.show()