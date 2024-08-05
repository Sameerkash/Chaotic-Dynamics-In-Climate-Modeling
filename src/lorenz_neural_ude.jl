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

data_pred = predict_neuralode(res2.u)
# Extend the timespan for prediction
extended_tspan = (t[end], 12.0)
extended_datasize = 12
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

plot_size = (1200, 600) # Width x Height in pixels
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px

# Plot losses over time
plot(tsteps, losses, title="Losses Over Time", xlabel="Time", ylabel="Loss", lw=3, legend=false, seriestype=:line, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)

# Use the trained parameters to make a prediction over the extended timespan
extended_prediction = extended_predict(p_trained, u0, extended_tsteps)

# Define the extended ODE problem for the true Lorenz system
extended_ode_prob = ODEProblem(lorenz!, u0, extended_tspan, p_)

# Solve the true Lorenz system over the extended timespan
extended_solution = Array(solve(extended_ode_prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=extended_tsteps))

plot()
plot!(t, ode_data[1, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u1", color=:red, alpha=0.3, xlabel="Time", ylabel="u", title="Lorenz System: Neural UDE Forecast", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)
plot!(t, data_pred[1, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u1", color=:red)

# Plot u2 - original data
plot!(t, ode_data[2, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u2", color=:green, alpha=0.3)
plot!(t, data_pred[2, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u2", color=:green)

# Plot u3 - original data
plot!(t, ode_data[3, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u3", color=:blue, alpha=0.3)
plot!(t, data_pred[3, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u3", color=:blue)


combined_pred_u1 = vcat(data_pred[1, :], extended_prediction[1, 2:end])
combined_pred_u2 = vcat(data_pred[2, :], extended_prediction[2, 2:end])
combined_pred_u3 = vcat(data_pred[3, :], extended_prediction[3, 2:end])

combined_tsteps = vcat(t, extended_tsteps[2:end])
# Plot the combined predictions
plot!(combined_tsteps, combined_pred_u1, seriestype=:line, linestyle=:dash, lw=2, label="Forecast u1", color=:red)
plot!(combined_tsteps, combined_pred_u2, seriestype=:line, linestyle=:dash, lw=2, label="Forecast u2", color=:green)
plot!(combined_tsteps, combined_pred_u3, seriestype=:line, linestyle=:dash, lw=2, label="Forecast u3", color=:blue)

# Plot the true ODE data and the extended prediction
plot!(extended_tsteps, extended_solution[1, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label=" True u1 (Extended)", color=:red)
plot!(extended_tsteps, extended_solution[2, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label=" True u2 (Extended)", color=:green)
plot!(extended_tsteps, extended_solution[3, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label=" True u3 (Extended)", color=:blue)

# Add legend and grid for the combined plot
plot!(legend=:outertopright, grid=true, legendfontsize=10)


# Calculate the actual terms from the Lorenz dynamics
function actual_dynamics!(du, u, p, t)
    lorenz!(du, u, p, t)
end

# Define the function to compute UDE missing terms
function ude_missing_terms!(du, u, p, t)
    û = U(u, p, st)[1]
    du[1] = u[2] - û[1]
    du[2] = -u[2] + 0.1 * û[2]
    du[3] = -u[3] + 10 * û[3]
end

# Solve the actual dynamics to get the true terms
actual_terms = Array(solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=tsteps))

# Solve the UDE to get the missing terms
missing_terms_prob = ODEProblem(ude_missing_terms!, u0, tspan, p)
missing_terms = Array(solve(missing_terms_prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=tsteps))

# Plot the missing terms compared to actual terms
# Plot u1 term
plot(tsteps, actual_terms[1, :], seriestype=:scatter, marker=:square, markersize=10.0, label="Actual u1 Term", color=:red, xlabel="Time", ylabel="Term", title="Lorenz System: UDE Recovered Terms", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)
plot!(tsteps, missing_terms[1, :], seriestype=:line, lw=2, linestyle=:dash, label="UDE Missing u1 Term", color=:red)

# Plot u2 term
plot!(tsteps, actual_terms[2, :], seriestype=:scatter, marker=:square, markersize=10.0, label="Actual u2 Term", color=:green)
plot!(tsteps, missing_terms[2, :], seriestype=:line, lw=2, linestyle=:dash, label="UDE Missing u2 Term", color=:green)

# Plot u3 term
plot!(tsteps, actual_terms[3, :], seriestype=:scatter, marker=:square, markersize=10.0, label="Actual u3 Term", color=:blue)
plot!(tsteps, missing_terms[3, :], seriestype=:line, lw=2, linestyle=:dash, label="UDE Missing u3 Term", color=:blue)

# Add legend, grid, and increase legend size
plot!(legend=:outertopright, grid=true, legendfontsize=12)
