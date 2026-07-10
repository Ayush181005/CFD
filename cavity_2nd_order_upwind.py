import numpy as np
from matplotlib import pyplot as plt, cm

l = 2
nx = 21 # nx = ny
nt = 500 # number of time steps
nit = 50 # iteration for pressure
dx = l / (nx - 1) # dx = dx
dy = dx

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

def pressure_poisson(p,u,v):
    global dx, dy, nit

    pn = np.empty_like(p)
    pn = p.copy()

    # internal iteration for pressure correction
    for _ in range(nit): # we iterate for pressure to get more accurate steadx state result
        pn = p.copy()

        du_dx = (3*u[2:-2, 2:-2] - 4*u[1:-3, 2:-2] + u[0:-4, 2:-2])/dx
        dv_dy = (3*v[2:-2, 2:-2] - 4*v[2:-2, 1:-3] + v[2:-2, 0:-4])/dy
        dudv_dxdy = (3*u[2:-2,2:-2] - 4*u[1:-3, 2:-2] + u[0:-4, 2:-2])*(3*v[2:-2, 2:-2] - 4*v[2:-2, 1:-3] + v[2:-2, 0:-4])/(dx*dy)

        p[2:-2,2:-2] = ((dy**2)*(pn[1:-3, 2:-2] + pn[3:-1, 2:-2]) + (dx**2)*(pn[2:-2, 1:-3] + pn[2:-2, 3:-1]) + rho*((dx**2)*(dy**2))*(du_dx**2 + dv_dy**2 + 2*dudv_dxdy))/(2*(dx**2 + dy**2))

        p[:,-1] = p[:,-2] # dp/dx = 0 at x = 2
        p[0,:] = p[1,:] # dp/dx = 0 at y = 0
        p[:,0] = p[:,1] # dp/dx = 0 at x = 0
        p[-1,:] = 1

    return p

def cavity_flow(u, v, p):
    global rho, dt, nt, dx, dy, nu

    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((nx, nx))

    for _ in range(nt):
        un = u.copy()
        vn = v.copy()

        p = pressure_poisson(p,u,v)

        # if u>0: # to setup upwind scheme
        udu_dx = un[2:-2, 2:-2] * (3*un[2:-2, 2:-2] - 4*un[1:-3, 2:-2] + un[0:-4, 2:-2])/(2*dx)
        vdu_dy = vn[2:-2, 2:-2] * (3*un[2:-2, 2:-2] - 4*un[2:-2, 1:-3] + un[2:-2, 0:-4])/(2*dy)
        dp_dx = (p[3:-1, 2:-2] - p[1:-3, 2:-2])/(2*dx)
        d2u_dx2 = (un[1:-3, 2:-2] - 2*un[2:-2, 2:-2] + un[3:-1, 2:-2])/(dx**2)
        d2u_dy2 = (un[2:-2, 1:-3] - 2*un[2:-2, 2:-2] + un[2:-2, 3:-1])/(dy**2)

        udv_dx = un[2:-2, 2:-2] * (3*vn[2:-2, 2:-2] - 4*vn[1:-3, 2:-2] + vn[0:-4, 2:-2])/(2*dx)
        vdv_dy = vn[2:-2, 2:-2] * (3*vn[2:-2, 2:-2] - 4*vn[2:-2, 1:-3] + vn[2:-2, 0:-4])/(2*dy)
        dp_dy = (p[2:-2, 3:-1] - p[2:-2, 1:-3])/(2*dy)
        d2v_dx2 = (vn[1:-3, 2:-2] - 2*vn[2:-2, 2:-2] + vn[3:-1, 2:-2])/(dx**2)
        d2v_dy2 = (vn[2:-2, 1:-3] - 2*vn[2:-2, 2:-2] + vn[2:-2, 3:-1])/(dy**2)

        u[2:-2, 2:-2] = un[2:-2, 2:-2] + dt*(nu*(d2u_dx2 + d2u_dy2) - (dp_dx/rho) - udu_dx - vdu_dy)
        v[2:-2, 2:-2] = vn[2:-2, 2:-2] + dt*(nu*(d2v_dx2 + d2v_dy2) - (dp_dy/rho) - udv_dx - vdv_dy)

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