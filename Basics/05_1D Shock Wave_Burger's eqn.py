# we'll model equation of shockwave and with symmetric boundary conditions
import numpy as np
from matplotlib import pyplot as plt
import sympy
from sympy.utilities.lambdify import lambdify # To convert sympy expression to a python function

x, nu, t = sympy.symbols('x nu t')
phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) + sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
phi_diff = phi.diff(x) # Differentiate: d phi / d x

u = -2 * nu * (phi_diff / phi) + 4 # Shockwave equation
ufunc = lambdify((t, x, nu), u)
# Now we can call it like ufunc(1,2,3) to get the value of u at t=1, x=2, nu=3

# variable declarations
l = 2*np.pi
nx = 101
nt = 100
dx = l / (nx - 1)
nu = .07
courant_number = .5
dt = courant_number * dx * nu

x = np.linspace(0, l, nx)
t = 0
u = np.asarray([ufunc(t, x0, nu) for x0 in x])

# Initial u
plt.plot(x, u)
# plt.show()


# We have a periodic boundary condition => u(0) = u(2*pi)

un = np.empty(nx)
for n in range(nt):
    un = u.copy()
    for i in range(1, nx-1):
        u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])
    
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
    u[-1] = u[0]

plt.plot(x, u)
plt.show()

u_analytical = np.asarray([ufunc(nt * dt, xi, nu) for xi in x])