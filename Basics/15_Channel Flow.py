import numpy
from matplotlib import pyplot, cm

# Variables
l = 0.5
nx = 17
dx = l / (nx - 1)
nt = 100
nit = 50
rho = 1
nu = 0.1
courant = 0.1
dt = courant * dx**2 / nu
F = 1

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, nx)
X, Y = numpy.meshgrid(x, y)

# NOTE that the boundary conditions set in the jupyter notebook are the periodic boundary conditions

def build_up_b(u,v):
    global rho, dt, dx

    b = numpy.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dx)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dx))**2))

    return b

def pressure_poisson(p,b):
    global dx, nit

    pn = numpy.empty_like(p)

    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dx**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) / (2 * (dx**2 + dx**2)) - dx**2 * dx**2 / (2 * (dx**2 + dx**2)) * b[1:-1, 1:-1])

        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        # opening boundary conditions, pressure
        p[:,0] = p[:,1]    # dp/dx = 0 at x = 0
        p[:,-1] = 1        # p = 1 (atm) at x = 2
    
    return p

def channel_flow(u,v,p):
    global nx, dx, dt, nu, nt

    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((nx, nx))

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(u, v)
        p = pressure_poisson(p, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dx) * (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dx**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0,:] = 0
        u[-1,:] = 0
        u[:,0] = 1
        u[:,-1] = u[:,-2]

        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = v[:,-2]

    return u, v, p

u = numpy.zeros((nx, nx))
v = numpy.zeros((nx, nx))
p = numpy.zeros((nx, nx))

u, v, p = channel_flow(u, v, p)

fig = pyplot.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
pyplot.colorbar()
# plotting the pressure field outlines
pyplot.contour(X, Y, p, cmap=cm.viridis)  
# plotting velocity field
# pyplot.streamplot(X,Y,u,v)
pyplot.quiver(X,Y,u,v)
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.title('2D Channel Flow')
pyplot.show()