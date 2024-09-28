using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, DifferentialEquations, Statistics
using Plots.PlotMeasures
gr()
# Intial Conditions u0 and constants p0

rng = Random.default_rng()
const S0 = 1.0
u0 = [S0 * 1, 0.0, 0.0]
p0 = Float64[
    10.0, # τSI
    28, # τIR
    8/3 # τID
]

# Range for training Data
tspan = (0.0, 30)
datasize = 30
t = range(tspan[1], tspan[2], length=datasize)


# Lorenz ODE
function Chaos!(du, u, p, t)
    (σ, ρ, β) = abs.(p)
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

prob = ODEProblem(Chaos!, u0, tspan, p0)
# True Solution of Solved ODE
ode_data = Array(solve(prob, Tsit5(), u0=u0, p=p0, saveat=t))

# Definition of Neural Network with activation function and layers.
activation = sigmoid
dudt2 = Lux.Chain(
    Lux.Dense(3, 25, activation),
    Lux.Dense(25, 3))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=t)


# Prediction function called in iterations
function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end


# Function defined for minimizing cost/loss of the solution
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
losses = Float32[]

# Call back to print iteration
callback5 = function (p, l, pred; doplot=true)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end

    # plot current prediction against data
    if doplot && length(losses) % 1000 == 0
        plt = scatter(t, ode_data[1, :], label="data")
        scatter!(plt, t, pred[1, :], label="prediction")
        display(plot(plt))
    end
    return false
end


pinit = ComponentArray(p)
callback5(pinit, loss_neuralode(pinit)...; doplot=true)

# Using Optimization.jl to solve the Neural ODE problem
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

# Using Adam Optimizer
result_neuralode = Optimization.solve(optprob, ADAM(0.01), callback=callback5, maxiters=50000)

optprob2 = remake(optprob, u0=result_neuralode.u)
result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.1),
    callback=callback5,
    allow_f_increases=false)
data_pred = predict_neuralode(result_neuralode2.u)



plot_size = (1200, 600)# Width x Height in pixels
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px

# plot losses
plot(t, losses, title="Losses Over Time", xlabel="Time", ylabel="Loss", lw=3, legend=false, seriestype=:line, size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)


# plot()

# # Plot u1
# plot!(t, ode_data[1, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u1", color=:red, alpha=0.3, xlabel="Time", ylabel="u", title="Lorenz System: Neural ODE", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)


# plot!(t, data_pred[1, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u1", color=:red)

# # Plot u2
# plot!(t, ode_data[2, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u2", color=:green, alpha=0.3)
# plot!(t, data_pred[2, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u2", color=:green)

# # Plot u3
# plot!(t, ode_data[3, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u3", color=:blue, alpha=0.3)
# plot!(t, data_pred[3, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u3", color=:blue)

# # Add legend and grid
# plot!(legend=:outertopright, grid=true, legendfontsize=14)


# # Extend the timespan for prediction
# extended_tspan = (0.0, 15.0)
# extended_datasize = 10
# extended_tsteps = range(extended_tspan[1], extended_tspan[2], length=extended_datasize)

# # Define the extended ODE problem with the trained parameters
# prob_neuralode_extended = NeuralODE(dudt2, extended_tspan, Tsit5(), saveat=extended_tsteps)

# # Use the trained parameters to make a prediction over the extended timespan
# extended_prediction = predict_neuralode(result_neuralode2.u)

# # Define the extended ODE problem for the true Lorenz system
# extended_ode_prob = ODEProblem(Chaos!, u0, extended_tspan, p0)

# # Solve the true Lorenz system over the extended timespan
# extended_solution = Array(solve(extended_ode_prob, Tsit5(), saveat=extended_tsteps))

# plot()

# # Plot the true ODE data and the extended prediction
# plot(extended_tsteps, extended_solution[1, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label="True u1", color=:red, xlabel="Time", ylabel="u", title="Lorenz System: Neural ODE Forecast", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)
# plot!(extended_tsteps, extended_prediction[1, :], lw=3, markersize=6.0, seriestype=:line, linestyle=:dash, label="Predicted u1", color=:red)

# plot!(extended_tsteps, extended_solution[2, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label="True u2", color=:green)
# plot!(extended_tsteps, extended_prediction[2, :], lw=3, markersize=6.0, seriestype=:line, linestyle=:dash, label="Predicted u2", color=:green)

# plot!(extended_tsteps, extended_solution[3, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label="True u3", color=:blue)
# plot!(extended_tsteps, extended_prediction[3, :], lw=3, markersize=6.0, seriestype=:line, linestyle=:dash, label="Predicted u3", color=:blue)

# # Add legend and grid
# plot!(legend=:outertopright, grid=true, legendfontsize=14)

plot(xtickfont=font("Times New Roman", 16),
ytickfont=font("Times New Roman", 16),
guidefont=font("Times New Roman", 18))
plot!(t, ode_data[1, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u1", color=:red, alpha=0.3, xlabel="Time", ylabel="u", title="Lorenz System: Neural ODE Forecast", size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)
plot!(t, data_pred[1, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u1", color=:red)

# Plot u2 - original data
plot!(t, ode_data[2, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u2", color=:green, alpha=0.3)
plot!(t, data_pred[2, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u2", color=:green)

# Plot u3 - original data
plot!(t, ode_data[3, :], seriestype=:scatter, marker=:circle, markersize=10.0, label="True u3", color=:blue, alpha=0.3)
plot!(t, data_pred[3, :], seriestype=:line, lw=2, markersize=5.0, label="Predicted u3", color=:blue)

# Add legend and grid for the original data plot
plot!(legend=:outertopright, grid=true, legendfontsize=14)

# Extend the timespan for prediction
extended_tspan = (t[end], 100.0)  # Start from the end of t
extended_datasize = 30
extended_tsteps = range(extended_tspan[1], extended_tspan[2], length=extended_datasize)

# Define the extended ODE problem with the trained parameters
prob_neuralode_extended = NeuralODE(dudt2, extended_tspan, Tsit5(), saveat=extended_tsteps)

# Use the trained parameters to make a prediction over the extended timespan
extended_prediction = predict_neuralode(result_neuralode2.u)

# Define the extended ODE problem for the true Lorenz system
extended_ode_prob = ODEProblem(Chaos!, u0, extended_tspan, p0)

# Solve the true Lorenz system over the extended timespan
extended_solution = Array(solve(extended_ode_prob, Tsit5(), saveat=extended_tsteps))

# Combine data_pred and extended_prediction for the plot
combined_pred_u1 = vcat(data_pred[1, :], extended_prediction[1, 2:end])
combined_pred_u2 = vcat(data_pred[2, :], extended_prediction[2, 2:end])
combined_pred_u3 = vcat(data_pred[3, :], extended_prediction[3, 2:end])

combined_tsteps = vcat(t, extended_tsteps[2:end])

# Plot the combined predictions
plot!(combined_tsteps, combined_pred_u1, seriestype=:line, linestyle=:dash, lw=2, label="Forecast u1", color=:red)
plot!(combined_tsteps, combined_pred_u2, seriestype=:line, linestyle=:dash, lw=2, label="Forecast u2", color=:green)
plot!(combined_tsteps, combined_pred_u3, seriestype=:line, linestyle=:dash, lw=2, label="Forecast u3", color=:blue)

# Plot the true ODE data and the extended prediction
plot!(extended_tsteps, extended_solution[1, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label="True u1 (Extended)", color=:red)
plot!(extended_tsteps, extended_solution[2, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label="True u2 (Extended)", color=:green)
plot!(extended_tsteps, extended_solution[3, :], lw=2, marker=:diamond, markersize=8.0, seriestype=:scatter, label="True u3 (Extended)", color=:blue)

# Add legend and grid for the combined plot
plot!(legend=:outertopright, grid=true, legendfontsize=10)