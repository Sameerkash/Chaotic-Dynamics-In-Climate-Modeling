using OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, Plots, StableRNGs

using Plots.PlotMeasures
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)

# Define the Lorenz system
function lorenz!(du, u, p, t)
    sigma, ρ, β = p
    du[1] = sigma * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

sigma = 10.0
ρ = 28.0
β = 8 / 3
p_ = [sigma, ρ, β]

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 8.0)
datasize = 8
tsteps = range(tspan[1], tspan[2], length=datasize)

prob = ODEProblem(lorenz!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=tsteps)
ode_data = Array(solution)
X = Array(solution)
t = solution.t

x̄ = mean(X, dims=2)
noise_magnitude = 5e-3
Xₙ = X

# Define the neural network
U = Lux.Chain(Lux.Dense(3, 25, sigmoid), Lux.Dense(25, 3))
p, st = Lux.setup(rng, U)
_st = st

# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, st)[1]
    du[1] = p_true[1] * (u[2] - û[1])
    du[2] = -u[2] + 0.1 * û[2]
    du[3] = -p_true[3] * u[3] + 10 * û[3]
end

nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

function predict(θ, X=Xₙ[:, 1], T=t)
    _prob = remake(prob_nn, u0=X, tspan=(T[1], T[end]), p=θ)
    Array(solve(_prob, Vern7(), saveat=T,
        abstol=1e-6, reltol=1e-6,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂)
end

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=50000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=5000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

p_trained = res2.minimizer

# Use the trained parameters to make a prediction over the original timespan
data_pred = predict(p_trained)

# Calculate the ideal unknown interactions of the predictor
# Compute the true interactions for the Lorenz system
Ȳ = [-p_[2] * (data_pred[1, :] .* data_pred[2, :])'; p_[3] * (data_pred[1, :] .* data_pred[2, :])']

# Get the neural network's guess of the interactions
pred = U(data_pred, p_trained, st)[1]

# Define the time steps for plotting
ts = tsteps

# Plot the true interactions and the UDE approximations
pl_reconstruction = plot(ts, pred', xlabel="Time", ylabel="U(x,y)", color=:red,
                         label=["UDE Approximation 1" "UDE Approximation 2" "UDE Approximation 3"])
plot!(ts, Ȳ', color=:black, label=["True Interaction 1" "True Interaction 2"])

# Show the plot
display(pl_reconstruction)