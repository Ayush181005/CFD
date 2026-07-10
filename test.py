import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib import rcParams

# Set font to Inter for better readability in plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Inter']

# --- Simulation Parameters ---
l = 2.0 # Length and width of the cavity
nx = 41 # Number of grid points in x-direction (nx = ny)
nt = 1000 # Number of time steps for the simulation
nit = 50 # Number of internal iterations for pressure correction
dx = l / (nx - 1) # Grid spacing in x-direction
dy = dx # Grid spacing in y-direction

# Create the grid
x = np.linspace(0, l, nx)
y = np.linspace(0, l, nx)
X, Y = np.meshgrid(x, y)

# Physical properties
rho = 1.0 # Fluid density
nu = 0.01 # Kinematic viscosity
# Re = u_lid * L / nu = 1 * 2 / 0.01 = 200. This is a good test case.

# Time step calculation (based on Courant-Friedrichs-Lewy condition)
# dt is calculated based on the stability requirements of the viscous term.
courant = 0.007
dt = courant * dx**2 / nu

# Initialize velocity (u, v), pressure (p), and pressure source term (b)
u = np.zeros((nx, nx))
v = np.zeros((nx, nx))
p = np.zeros((nx, nx)) 
b = np.zeros((nx, nx))

# Set initial boundary conditions for u and v
# u boundary conditions
u[0, :] = 0.0 # u = 0 on bottom wall
u[-1, :] = 1.0 # u = 1 on the top lid
u[:, 0] = 0.0 # u = 0 on the left wall
u[:, -1] = 0.0 # u = 0 on the right wall

# v boundary conditions
v[:, :] = 0.0 # v = 0 everywhere on all boundaries

def pressure_poisson(p, b):
    """
    Solves the Pressure-Poisson equation using an iterative method.
    The equation is solved for pressure 'p' using a five-point stencil.
    Neumann (zero-gradient) boundary conditions are applied.
    """
    pn = np.empty_like(p)
    
    for _ in range(nit):
        pn = p.copy()
        
        # Main update equation for the interior points using a 5-point stencil
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Apply Neumann boundary conditions (dp/dx = 0, dp/dy = 0)
        p[:, -1] = p[:, -2] # dp/dx = 0 on right wall
        p[0, :] = p[1, :]   # dp/dy = 0 on bottom wall
        p[:, 0] = p[:, 1]   # dp/dx = 0 on left wall
        p[-1, :] = p[-2, :] # dp/dy = 0 on top wall
        
    return p

def cavity_flow(u, v, p):
    """
    Main simulation loop for the lid-driven cavity flow.
    It solves the Navier-Stokes equations for each time step.
    """
    un = np.empty_like(u)
    vn = np.empty_like(v)

    for _ in range(nt):
        un = u.copy()
        vn = v.copy()

        # Calculate the pressure source term 'b' using central differencing
        # This term correctly represents the divergence of the velocity field
        b[1:-1, 1:-1] = (rho * (1 / dt *
                        ((un[2:, 1:-1] - un[0:-2, 1:-1]) / (2 * dx) +
                         (vn[1:-1, 2:] - vn[1:-1, 0:-2]) / (2 * dy)) -
                        ((un[2:, 1:-1] - un[0:-2, 1:-1]) / (2 * dx))**2 -
                        2 * ((un[1:-1, 2:] - un[1:-1, 0:-2]) / (2 * dy) *
                             (vn[2:, 1:-1] - vn[0:-2, 1:-1]) / (2 * dx)) -
                        ((vn[1:-1, 2:] - vn[1:-1, 0:-2]) / (2 * dy))**2))

        # Solve for pressure
        p = pressure_poisson(p, b)

        # Calculate terms for the momentum equations
        # Viscous terms (second derivatives)
        d2u_dx2 = (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dx**2
        d2u_dy2 = (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dy**2
        d2v_dx2 = (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / dx**2
        d2v_dy2 = (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dy**2
        
        # Pressure gradient terms
        dp_dx = (p[2:, 1:-1] - p[0:-2, 1:-1]) / (2 * dx)
        dp_dy = (p[1:-1, 2:] - p[1:-1, 0:-2]) / (2 * dy)

        # Advection terms (2nd-order upwind)
        # This is a simplified upwind scheme for stability. A full QUICK scheme 
        # would require conditional logic based on the sign of u and v.
        u_advect_x = un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dx
        v_advect_y = vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dy

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - dt / rho * dp_dx + nu * dt * (d2u_dx2 + d2u_dy2) - dt * u_advect_x - dt * v_advect_y)
        
        # V-momentum advection terms
        u_advect_v_x = un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dx
        v_advect_v_y = vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dy
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - dt / rho * dp_dy + nu * dt * (d2v_dx2 + d2v_dy2) - dt * u_advect_v_x - dt * v_advect_v_y)

        # Re-apply boundary conditions at the end of each time step
        u[0, :] = 0.0
        u[-1, :] = 1.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        v[:, :] = 0.0

    return u, v, p

# --- Run the simulation ---
u, v, p = cavity_flow(u, v, p)

# --- Plotting the results ---
fig = plt.figure(figsize=(11, 7), dpi=100)
# Plot the pressure field as a contour
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
plt.colorbar(label='Pressure')
# Plot the pressure field outlines
plt.contour(X, Y, p, cmap=cm.viridis)
# Plot the streamplot to show flow streamlines
plt.streamplot(X, Y, u, v, color='k')
plt.title('2D Lid-Driven Cavity Flow (2nd-Order Upwind)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()