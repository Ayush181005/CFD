import numpy as np
from matplotlib import pyplot as plt, cm

l = 2
nx = 41 # nx = ny
nt = 500 # number of time steps
nit = 50 # iteration for pressure
dx = 2 / (nx - 1) # dx = dy

x = np.linspace(0, l, nx)
y = np.linspace(0, l, nx)
X, Y = np.meshgrid(x, y)

rho = 1
nu = 0.8
alpha = 2
courant = 0.005
dt = courant * dx / nu

u = np.zeros((nx, nx))
v = np.zeros((nx, nx))
p = np.zeros((nx, nx)) 
b = np.zeros((nx, nx))
Temp = np.ones((nx, nx))*40

un = np.empty_like(u)
vn = np.empty_like(v)

# to reduce errors, we will solve in different steps
def buildup_b(b,u,v):
    global rho, dt, dx

    # b comes from the RHS of the poisson's equation
    b[1:-1,1:-1] = (rho/(2*dx)) * ((1/dt)*(u[1:-1,2:] - u[1:-1,0:-2] + v[2:,1:-1] - v[0:-2,1:-1]) - (1/(2*dx))*((u[2:,1:-1]-u[0:-2,1:-1])**2 + (v[1:-1,2:]-v[1:-1,0:-2])**2 + (2*(u[2:,1:-1:]-u[0:-2,1:-1])*(v[1:-1,2:]-v[1:-1,0:-2]))))

    return b

def pressure_poisson(p,b):
    global dx, nit

    pn = np.empty_like(p)
    pn = p.copy()

    # internal iteration for pressure correction
    for _ in range(nit): # we iterate for pressure to get more accurate steady state result
        pn = p.copy()

        p[1:-1,1:-1] = (pn[2:,1:-1] + pn[0:-2,1:-1] + pn[1:-1,2:] + pn[1:-1,0:-2] - (dx**2)*b[1:-1,1:-1])/4

        p[:,-1] = p[:,-2] # dp/dx = 0 at x = 2
        p[0,:] = p[1,:] # dp/dy = 0 at y = 0
        p[:,0] = p[:,1] # dp/dx = 0 at x = 0
        p[-1,:] = 1

    return p

def cavity_flow(u, v, p, Temp):
    global rho, dt, nt, dx, nu

    un = np.empty_like(u)
    vn = np.empty_like(v)
    Tn = np.empty_like(Temp)
    b = np.zeros((nx, nx))

    for _ in range(nt):
        un = u.copy()
        vn = v.copy()
        Tn = Temp.copy()

        b = buildup_b(b,u,v)
        p = pressure_poisson(p,b)

        u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt/dx**2*((nu*(un[1:-1,2:]-4*un[1:-1,1:-1]+un[1:-1,0:-2]+un[2:,1:-1]+un[0:-2, 1:-1])) - dx*(un[1:-1, 1:-1] * (un[2:, 1:-1] -un[1:-1, 1:-1]) + vn[1:-1, 1:-1] * (un[1:-1, 2:] - un[1:-1, 1:-1]) + (p[1:-1, 2:] - p[1:-1, 0:-2])/(2*rho)))

        v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt/dx**2*((nu*(vn[1:-1,2:]-4*vn[1:-1,1:-1]+vn[1:-1,0:-2]+vn[2:,1:-1]+vn[0:-2, 1:-1])) - dx*(un[1:-1, 1:-1] * (vn[2:, 1:-1] -vn[0:-2, 1:-1]) + vn[1:-1, 1:-1] * (vn[1:-1, 2:] - vn[1:-1, 0:-2]) + (p[2:, 1:-1] - p[0:-2, 1:-1])/(2*rho)))

        Temp[1:-1,1:-1] = Tn[1:-1,1:-1] + dt*(alpha*(Tn[2:,1:-1]+Tn[0:-2,1:-1]+Tn[1:-1,2:]+Tn[1:-1,0:-2]-4*Tn[1:-1,1:-1])/dx**2 - un[1:-1,1:-1]*(Tn[2:,1:-1]-Tn[0:-2,1:-1])/(2*dx) - vn[1:-1,1:-1]*(Tn[1:-1,2:]-Tn[1:-1,0:-2])/(2*dx))

        u[0,:] = 0
        u[-1,:] = 1 # set velocity on cavity lid equal to 1
        u[:,0] = 0
        u[:,-1] = 0

        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0

        Temp[0,:] = Temp[1,:]
        Temp[-1,:] = 35
        Temp[:,0] = Temp[:,1]
        Temp[:,-2] = Temp[:,-1]

    return u, v, p, Temp

u, v, p, Temp = cavity_flow(u, v, p, Temp)

fig = plt.figure(figsize=(11,7), dpi=100)
# plotting the pressure/temperature field as a contour
plt.contourf(X, Y, Temp, cmap=cm.seismic)
# plt.contourf(X, Y, p, cmap=cm.viridis)
plt.colorbar()
# plt.contour(X, Y, Temp, cmap=cm.seismic)
# plotting velocity field
plt.streamplot(X,Y,u,v,color="black")
# plt.quiver(X,Y,u,v)
plt.title('Cavity Flow')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()