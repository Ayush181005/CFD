import numpy as np
import matplotlib.pyplot as plt

# constants for this function
L = 1 # length of the string
a = 1
totalT = 5 # total time to simulate

n = 25 # number of terms to include from the series solution
nx = 50 # number of points to plot
nt = 100 # number of points to plot

x = np.linspace(0, L, nx)
t = np.linspace(0, totalT, nt)
u = np.zeros(nx)

def calculate_y(xi, ti):
    global u, n, L, a
    u = np.zeros(nx)
    for i in range(1, n-1):
        u += (6/(i*np.pi*L))*(2 + ((2/i*np.pi)*np.sin(i*np.pi/3)) + (((-1)**i)*(1 - (1/(i*np.pi*L)) + ((3*3**0.5)/(2*i*np.pi*L)))))*np.sin(i*np.pi*xi/L)*np.cos(i*np.pi*a*ti/L)

# print(u)

# Plot the solution
X, T = np.meshgrid(x, t)
U = np.zeros((nt, nx))

for i in range(nt):
    calculate_y(x, t[i])
    U[i, :] = u
    u = np.zeros(nx)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel("u(x, t)")
plt.title(r"Wave Equation PDE: $a^2\frac{\partial^2u}{\partial x^2} = \frac{\partial^2u}{\partial t^2}$")
plt.show()