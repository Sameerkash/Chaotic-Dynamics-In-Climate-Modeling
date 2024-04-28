include("lorenz_ode.jl")

using .LorenzODE


u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 100.0)

sol = LorenzODE.Lorenz(u0, tspan)