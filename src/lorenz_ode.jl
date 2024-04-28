module LorenzODE

export Lorenz, LorenzPlotSolution

using DifferentialEquations
using Plots


function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end


function Lorenz(u0, tspan, tsteps)
    σ = 10.0
    ρ = 28.0
    β = 8 / 3
    p = [σ, ρ, β]

    prob = ODEProblem(lorenz!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps)
    return sol
end

function LorenzPlotSolution(sol)
    plot(sol, vars=(1, 2, 3), xlabel="x", ylabel="y", zlabel="z", title="Lorenz Attractor")
end

end


