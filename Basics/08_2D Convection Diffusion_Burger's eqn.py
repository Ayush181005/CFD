import numpy as np
from matplotlib import pyplot as plt, cm
# For animation
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML

# Variables
l = 2
nx = 41 # nx = ny
dx = l/(nx-1) # dx = dy
# nt = 500 # This time we'll use tolerance to stop the simulation, not nt
nu = 0.3
courant = 0.2
dt = courant * dx**2 / nu
tolerance = 1e-5 # threshold for convergence

# Generating the mesh
x = np.linspace(0, l, nx)
y = np.linspace(0, l, nx)

# Initialization
u = np.ones((nx, nx))
v = np.ones((nx, nx))
un = np.ones((nx, nx))
vn = np.ones((nx, nx))
# Intital and Boundary Condition
u[int(.5/dx):int(1/dx+1),int(.5/dx):int(1/dx+1)]=2
v[int(.5/dx):int(1/dx+1),int(.5/dx):int(1/dx+1)]=4

# Function to update the field
def conv_diffuse (u, v):
    global dt, dx, nu

    un = u.copy()
    vn = v.copy()

    # Discretization eqn using array slicing
    # u[1:nx-1,1:nx-1] = un[1:nx-1,1:nx-1] + (dt / dx**2) * (nu * (un[2:nx,1:nx-1] + un[0:nx-2,1:nx-1] + un[1:nx-1,2:nx] + un[1:nx-1,0:nx-2] - 4*un[1:nx-1,1:nx-1]) - dx*(un[1:nx-1,1:nx-1] * (un[2:nx,1:nx-1] - un[1:nx-1,1:nx-1]) + vn[1:nx-1,1:nx-1] * (un[1:nx-1,2:nx] - un[1:nx-1,1:nx-1])))
    u[1:-1,1:-1] = un[1:-1,1:-1] + (dt / dx**2) * (nu * (un[1:-1,2:] + un[1:-1,0:-2] + un[2:,1:-1] + un[0:-2,1:-1] - 4*un[1:-1,1:-1]) - dx*(un[1:-1,1:-1] * (un[2:,1:-1] - un[1:-1,1:-1]) + vn[1:-1,1:-1] * (un[1:-1,2:] - un[1:-1,1:-1])))

    v[1:-1,1:-1] = vn[1:-1,1:-1] + (dt / dx**2) * (nu * (vn[1:-1,2:] + vn[1:-1,0:-2] + vn[2:,1:-1] + vn[0:-2,1:-1] - 4*vn[1:-1,1:-1]) - dx*(un[1:-1,1:-1] * (vn[2:,1:-1] - vn[1:-1,1:-1]) + vn[1:-1,1:-1] * (vn[1:-1,2:] - vn[1:-1,1:-1])))

    # Boundary Conditions
    u[0,:] = 1 # Bottom Boundary
    u[-1,:] = 1 # Top Boundary
    u[:,0] = 1 # Left Boundary
    u[:,-1] = 1 # Right Boundary

    v[0,:] = 1 # Bottom Boundary
    v[-1,:] = 1 # Top Boundary
    v[:,0] = 1 # Left Boundary
    v[:,-1] = 1 # Right Boundary

    return u,v

# Post Processing
velocity = np.sqrt(u**2 + v**2)
fig, ax = plt.subplots(figsize=(8, 6))
X, Y = np.meshgrid(x, y) # Grid
contour = ax.contourf(X, Y, velocity, alpha=0.75, cmap=cm.viridis) # Intital Contour Plot
cbar = fig.colorbar(contour)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('2D Contour Plot of Convection-Diffused Field')

def update(frame):
    global u, v, un, vn, tolerance, ani
    
    un = u.copy()
    vn = v.copy()

    u, v = conv_diffuse(u, v)
    
    velocity = np.sqrt(u**2 + v**2)

    ax.clear()
    contour = ax.contourf(X, Y, velocity, alpha=0.75, cmap=cm.viridis) # Intital Contour Plot
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'2D Contour Plot of Diffused Field (Frame{frame})')
    
    diff = np.linalg.norm(u-un) + np.linalg.norm(v-vn)
    if diff<tolerance:
        ani.event_source.stop()
    return ax.collections

# create the animation
ani = FuncAnimation(fig, update, frames=500, interval = 100, blit=True)
plt.show()
# HTML(ani.to_jshtml())