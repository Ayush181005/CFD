import numpy as np
from matplotlib import pyplot as plt, cm

lx = 2
ly = 2
nx = 21
ny = 21
nt = 500 # number of time steps
nit = 100 # iteration for pressure
dx = lx / (nx - 1)
dy = ly / (ny - 1)

x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
X, Y = np.meshgrid(x, y)

rho = 1.2
nu = 1
courant = 0.0005
dt = courant * min(dx, dy) / nu
# print(courant)

u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny)) 
b = np.zeros((nx, ny))

un = np.empty_like(u)
vn = np.empty_like(v)

# to reduce errors, we will solve in different steps
def buildup_b(b,u,v):
    global rho, dt, dx, dy

    # b comes from the RHS of the poisson's equation

    b[1:-1,1:-1] = dx**2*(-rho/(2*dx)) * ((1/dt)*(u[1:-1,2:] - u[1:-1,0:-2] + v[2:,1:-1] - v[0:-2,1:-1]) - (1/(2*dx))*((u[2:,1:-1]-u[0:-2,1:-1])**2 + (v[1:-1,2:]-v[1:-1,0:-2])**2 + (2*(u[2:,1:-1]-u[0:-2,1:-1])*(v[1:-1,2:]-v[1:-1,0:-2]))))

    return b

def pressure_poisson(p,b):
    global dx, dy, nit

    pn = np.empty_like(p)
    pn = p.copy()

    # internal iteration for pressure correction
    for _ in range(nit): # we iterate for pressure to get more accurate steady state result
        pn = p.copy()

        p[1:-1,1:-1] = (pn[2:,1:-1] + pn[0:-2,1:-1] + pn[1:-1,2:] + pn[1:-1,0:-2] + b[1:-1,1:-1])/4

        p[:,-1] = p[:,-2] # dp/dx = 0 at x = 2
        p[0,:] = p[1,:] # dp/dy = 0 at y = 0
        p[:,0] = p[:,1] # dp/dx = 0 at x = 0
        p[-1,:] = 1 # p = 0 at y = 2

    return p

def cavity_flow(u, v, p):
    global rho, dt, nt, dx, dy, nu

    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((nx, nx))

    for _ in range(nt):
        un = u.copy()
        vn = v.copy()

        b = buildup_b(b,u,v)
        p = pressure_poisson(p,b)

        u[1:-1,1:-1] = un[1:-1,1:-1] + dt*(nu*(((un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])/(dx**2)) + ((un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])/(dy**2))) - (un[1:-1,1:-1]*(un[2:,1:-1]-un[0:-2,1:-1])/(2*dx)) - (vn[1:-1,1:-1]*(un[1:-1,2:]-un[1:-1,0:-2])/(2*dy)) - ((p[1:-1,2:]-p[1:-1,0:-2])/(2*rho*dx)))

        v[1:-1,1:-1] = vn[1:-1,1:-1] + dt*(nu*(((vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])/(dx**2)) + ((vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])/(dy**2))) - (un[1:-1,1:-1]*(vn[2:,1:-1]-vn[0:-2,1:-1])/(2*dx)) - (vn[1:-1,1:-1]*(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy)) - ((p[2:,1:-1]-p[0:-2,1:-1])/(2*rho*dy)))

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
# plotting velocity field
# plt.streamplot(X,Y,u,v)
plt.quiver(X,Y,u,v)
plt.title('Cavity Flow')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()