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

shallow = true

# Define the neural network
U = Lux.Chain(Lux.Dense(3, 10, sigmoid), Lux.Dense(10, 20, sigmoid), Lux.Dense(20, 3))
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

shallowLosses = Float64[]
deepLosses = Float64[]


callback = function (p, l)
    if (shallow)
        push!(shallowLosses, l)
        if length(shallowLosses) % 50 == 0
            println("Current shallowloss after $(length(shallowLosses)) iterations: $(shallowLosses[end])")
        end
    else
        push!(deepLosses, l)
        if length(deepLosses) % 50 == 0
            println("Current deepLossesafter $(length(deepLosses)) iterations: $(deepLosses[end])")
        end
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=50000)
# println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=5000)
# println("Final training loss after $(length(losses)) iterations: $(losses[end])")

p_trained = res2.minimizer


epochs = 0:4999

plot_size = (1200, 600)# Width x Height in pixels
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px
# Plot losses

plot()
plot(title="Losses Over Epoch", xlabel="Epoch", ylabel="Loss", ylims=(-10, 100), xlims=(0, 5000), size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)
# Add the data series
plot!(epochs, shallowLosses, legend=false, seriestype=:line, linestyle=:dash, lw=2, label="Shallow Network", color=:red)  # Uncomment and provide data if needed
plot!(epochs, deepLosses, legend=false, seriestype=:line, lw=2, label="Deep Network", color=:blue)
plot!(legend=:outertopright, grid=true, legendfontsize=14)


# Extend the timespan for prediction
extended_tspan = (0.0, 15.0)
extended_datasize = 15
extended_tsteps = range(extended_tspan[1], extended_tspan[2], length=extended_datasize)

# Define the extended ODE problem with the trained parameters
prob_node2 = ODEProblem(nn_dynamics!, u0, extended_tspan, p_trained)

# Define the extended prediction function
function extended_predict(θ, X=u0, T=extended_tsteps)
    _prob = remake(prob_node2, u0=X, tspan=(T[1], T[end]), p=θ)
    Array(solve(_prob, Vern7(), saveat=T,
        abstol=1e-6, reltol=1e-6,
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# Use the trained parameters to make a prediction over the extended timespan
extended_prediction = extended_predict(p_trained, u0, extended_tsteps)

# Define the extended ODE problem for the true Lorenz system
extended_ode_prob = ODEProblem(lorenz!, u0, extended_tspan, p_)

# Solve the true Lorenz system over the extended timespan
extended_solution = Array(solve(extended_ode_prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=extended_tsteps))

# Plot the true ODE data with dots and the extended prediction with lines
# Different colors for u1, u2, u3, with legends and axis titles

# Plot u1
plot(extended_tsteps, extended_solution[1, :], seriestype=:scatter, marker=:circle, markerSize=10.0, label="True u1", color=:red, xlabel="Time", ylabel="u", title="Lorenz System: Neural UDE Forecast")
plot!(extended_tsteps, extended_prediction[1, :], seriestype=:line, linestyle=:dash, lw=2, label="Predicted u1", color=:red)

# Plot u2
plot!(extended_tsteps, extended_solution[2, :], seriestype=:scatter, marker=:circle, markerSize=10.0, label="True u2", color=:green)
plot!(extended_tsteps, extended_prediction[2, :], seriestype=:line, linestyle=:dash, lw=2, label="Predicted u2", color=:green)

# Plot u3
plot!(extended_tsteps, extended_solution[3, :], seriestype=:scatter, marker=:circle, markerSize=10.0, label="True u3", color=:blue)
plot!(extended_tsteps, extended_prediction[3, :], seriestype=:line, linestyle=:dash, lw=2, label="Predicted u3", color=:blue)

# Add legend and grid
plot!(legend=:topleft, grid=true)
