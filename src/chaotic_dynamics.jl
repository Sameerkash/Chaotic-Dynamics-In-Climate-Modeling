include("lorenz_ode.jl")

using .LorenzODE

u0 = [1.0, 0.0, 0.0]

tspan = (0.0, 100.0)
datasize = 1000
t = range(tspan[1], tspan[2], length=datasize)

sol, p = LorenzODE.LorenzSolution(u0, tspan, t);

LorenzODE.LorenzPlotSolution(sol)