import numpy as np                        # here we load numpy
from matplotlib import pyplot as plt      # here we load matplotlib

# Initial
l = np.pi # total length
k = 1 # thermal diffusivity

nx = 81  # number of small elements, try changing this number from 41 to 81 and Run All ... what happens?
dx = l / (nx-1) # width of each element
nt = 1500    # number of times the simulations will run, is the number of timesteps we want to calculate
courant_number = 0.3
dt = courant_number * (dx**2) / k

# Intital Conditions
u = 100*np.ones(nx)

un = 100*np.ones(nx) # initialize a temporary array

plt.plot(np.linspace(0, l, nx), un)

for n in range(nt):  # loop for values of n from 0 to nt, so it will run nt times
    un = u.copy() # copy the existing values of u into un
    u[1:-1] = un[1:-1] + ((k*dt/dx**2)*(un[2:] -2*un[1:-1] + un[0:-2]))
    
    # Boundary Conditions
    u[0] = 0 # u(0, t) = 0
    u[-1] = 0 # u(l, t) = 0


plt.plot(np.linspace(0, l, nx), u)
plt.show()