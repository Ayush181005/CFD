import numpy as np                        # here we load numpy
from matplotlib import pyplot as plt      # here we load matplotlib

# Initial
l = 2 # total length
nx = 500  # number of small elements, try changing this number from 41 to 81 and Run All ... what happens?
dx = l / (nx-1) # width of each element
nt = 350    # number of times the simulations will run, is the number of timesteps we want to calculate
# dt = .005  # amount of time each timestep covers (delta t)
c = 1      # assume wavespeed of c = 1
disturbance_amplitude = 2
courant_number = 0.1
dt = courant_number * dx

# Intital Conditions
u = np.ones(nx)      #numpy function ones()

# Boundary Conditions (not actually)
u[int(.5 / dx):int(1 / dx + 1)] = disturbance_amplitude  # Creating Disturbance: setting u = 1.5 between 0.5 and 1 as per our I.C.s
# print(u)

# initial disturbance
plt.plot(np.linspace(0, l, nx), u)
# plt.show()

un = np.ones(nx) # initialize a temporary array
for n in range(nt):  # loop for values of n from 0 to nt, so it will run nt times
    un = u.copy() # copy the existing values of u into un
    for i in range(nx):
        u[i] = un[i] - c * dt * (un[i] - un[i-1]) / dx
    
    # Animating every iteration in plt
    # plt.cla()
    # plt.plot(np.linspace(0, l, nx), u)
    # plt.pause(0.05)

plt.plot(np.linspace(0, l, nx), u)
plt.show()