import numpy as np                        # here we load numpy
from matplotlib import pyplot as plt      # here we load matplotlib

# Initial
l = 5 # total length
nx = 81  # number of small elements, try changing this number from 41 to 81 and Run All ... what happens?
dx = l / (nx-1) # width of each element
nt = 500    # number of times the simulations will run, is the number of timesteps we want to calculate
nu = 0.3      # viscocity / diffusivity of nu = 0.3
disturbance_amplitude = 1.5
courant_number = 0.3
dt = courant_number * (dx**2) / nu

# Intital Conditions
u = np.ones(nx)      #numpy function ones()

# Boundary Conditions (not actually)
u[int(.5 / dx):int(1 / dx + 1)] = disturbance_amplitude  # Creating Disturbance: setting u = 1.5 between 0.5 and 1 as per our I.C.s

# initial disturbance
plt.plot(np.linspace(0, l, nx), u)
# plt.show()

un = np.ones(nx) # initialize a temporary array
convergence_count = 0 # For convergence condition

for n in range(nt):  # loop for values of n from 0 to nt, so it will run nt times
    un = u.copy() # copy the existing values of u into un
    for i in range(1, nx-1):
        u[i] = un[i] - un[i]*(dt/dx)*(un[i+1]-un[i]) + nu * (dt / (dx**2)) * (un[i+1] - (2*un[i]) + un[i-1])

        # Condition for convergence
        if abs(un[i] - u[i]) <= 0.00000000025 and n>1000:
            convergence_count += 1

    # Convergence should occur minimum 2 times
    if convergence_count >= 2:
        print("Converged at time step: ", n)
        print("Total time steps: ", nt)
        break

    plt.cla()
    plt.axis([0, l, 0.9, disturbance_amplitude+0.5])
    plt.plot(np.linspace(0, l, nx), u)
    plt.pause(0.01)

plt.plot(np.linspace(0, l, nx), u)
plt.show()