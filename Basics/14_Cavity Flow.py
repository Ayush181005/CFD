import numpy as np
from matplotlib import pyplot as plt, cm

l = 2
nx = 21 # nx = ny
nt = 500 # number of time steps
nit = 50 # iteration for pressure
dx = l / (nx - 1) # dx = dx

x = np.linspace(0, l, nx)
y = np.linspace(0, l, nx)
X, Y = np.meshgrid(x, y)

rho = 1.2
nu = 1
courant = 0.007
# dt = 0.001
# courant = nu*dt/dx
dt = courant * dx / nu
# print(courant)

u = np.zeros((nx, nx))
v = np.zeros((nx, nx))
p = np.zeros((nx, nx)) 
b = np.zeros((nx, nx))

un = np.empty_like(u)
vn = np.empty_like(v)

# to reduce errors, we will solve in different steps
def buildup_b(b,u,v):
    global rho, dt, dx

    # b comes from the RHS of the poisson's equation
    b[1:-1,1:-1] = (rho/(2*dx)) * ((1/dt)*(u[1:-1,2:] - u[1:-1,0:-2] + v[2:,1:-1] - v[0:-2,1:-1]) - (1/(2*dx))*((u[2:,1:-1]-u[0:-2,1:-1])**2 + (v[1:-1,2:]-v[1:-1,0:-2])**2 + (2*(u[2:,1:-1]-u[0:-2,1:-1])*(v[1:-1,2:]-v[1:-1,0:-2]))))

    return b

def pressure_poisson(p,b):
    global dx, nit

    pn = np.empty_like(p)
    pn = p.copy()

    # internal iteration for pressure correction
    for _ in range(nit): # we iterate for pressure to get more accurate steadx state result
        pn = p.copy()

        p[1:-1,1:-1] = (pn[2:,1:-1] + pn[0:-2,1:-1] + pn[1:-1,2:] + pn[1:-1,0:-2] - (dx**2)*b[1:-1,1:-1])/4

        p[:,-1] = p[:,-2] # dp/dx = 0 at x = 2
        p[0,:] = p[1,:] # dp/dx = 0 at y = 0
        p[:,0] = p[:,1] # dp/dx = 0 at x = 0
        p[-1,:] = 1

    return p

def cavity_flow(u, v, p):
    global rho, dt, nt, dx, nu

    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((nx, nx))

    for _ in range(nt):
        un = u.copy()
        vn = v.copy()

        b = buildup_b(b,u,v)
        p = pressure_poisson(p,b)

        u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt/dx**2*(nu*(un[1:-1,2:]-4*un[1:-1,1:-1]+un[1:-1,0:-2]+un[2:,1:-1]+un[0:-2, 1:-1]) - dx*(un[1:-1, 1:-1] * (un[2:, 1:-1] -un[1:-1, 1:-1]) + vn[1:-1, 1:-1] * (un[1:-1, 2:] - un[1:-1, 1:-1]) + (p[1:-1, 2:] - p[1:-1, 0:-2])/(2*rho)))

        v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt/dx**2*((nu*(vn[1:-1,2:]-4*vn[1:-1,1:-1]+vn[1:-1,0:-2]+vn[2:,1:-1]+vn[0:-2, 1:-1])) - dx*(un[1:-1, 1:-1] * (vn[2:, 1:-1] -vn[0:-2, 1:-1]) + vn[1:-1, 1:-1] * (vn[1:-1, 2:] - vn[1:-1, 0:-2]) + (p[2:, 1:-1] - p[0:-2, 1:-1])/(2*rho)))

        u[0,:] = 0
        u[-1,:] = 1 # set velocity on cavity lid equal to 1
        u[:,0] = 0
        u[:,-1] = 0

        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0

    return u, v, p

u, v, p = cavity_flow(u, v, p)

fig = plt.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
# plotting the pressure field outlines
plt.colorbar()
plt.contour(X, Y, p, cmap=cm.viridis)  
plt.streamplot(X,Y,u,v)
# plt.quiver(X,Y,u,v)
# plotting velocity field
plt.title('Cavity Flow')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()