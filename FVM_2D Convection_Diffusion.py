import numpy as np
from matplotlib import pyplot as plt, cm
# For animation
from matplotlib.animation import FuncAnimation

# Variables
lx = 2
ly = 2
nx = 41 # nx = ny
ny = 41 # nx = ny
dx = lx/(nx-1) # dx = dy
dy = ly/(ny-1) # dx = dy
nu = 0.3
courant = 0.2
dt = courant * dx**2 / nu
tolerance = 1e-5 # threshold for convergence

# Generating the mesh
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

# Initialization
u = np.ones((nx, ny))
v = np.ones((nx, ny))
un = np.ones((nx, ny))
vn = np.ones((nx, ny))
# Intital and Boundary Condition
u[int(.5/dx):int(1/dx+1),int(.5/dx):int(1/dx+1)]=2
v[int(.5/dx):int(1/dx+1),int(.5/dx):int(1/dx+1)]=4

# Function to update the field
def conv_diffuse (u, v):
    global dt, dx, dy, nu

    un = u.copy()
    vn = v.copy()

    u[1:-1,1:-1] = un[1:-1,1:-1] + dt*(nu*(((un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])/dx**2) + ((un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])/dy**2)) - (un[1:-1,1:-1]*(un[2:,1:-1]-un[0:-2,1:-1])/(2*dx)) - (vn[1:-1,1:-1]*(un[1:-1,2:]-un[1:-1,0:-2])/(2*dy)))

    v[1:-1,1:-1] = vn[1:-1,1:-1] + dt*(nu*(((vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])/dx**2) + ((vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])/dy**2)) - (un[1:-1,1:-1]*(vn[2:,1:-1]-vn[0:-2,1:-1])/(2*dx)) - (vn[1:-1,1:-1]*(vn[1:-1,2:]-vn[1:-1,0:-2])/(2*dy)))

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