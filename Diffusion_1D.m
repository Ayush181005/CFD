clear all
close all
clc

%% Defining Mesh
nx = 5;
l = 1;
dx = l/(nx - 1);

%% Initializing Problem
% d2u/dx2 = 0

% Boundary Conditions
u(1) = 0;
u(nx) = 1;

u_new(1) = 0;
u_new(nx) = 1;
% all other are set to zero automatically

%Error = sum(u_new - u), should be less than threshold
error_mag = 1;
convergence_threshold = 1e-6;
iterations = 0;

%% Calculations
while error_mag > convergence_threshold
    for i = 2:(nx-1)
        u_new(i) = 0.5.*(u(i-1) + u(i+1));
        iterations = iterations + 1;
    end

    % Calculating error
    error_mag = 0;
    for i = 2:(nx-1)
        error_mag = error_mag + mod(u_new(i),u(i));
    end

    u = u_new;
end

%% Post Processing
x_domain = ((1:nx)-1).*dx;
figure; plot(x_domain, u)