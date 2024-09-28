using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, DifferentialEquations, Statistics
rng = Random.default_rng()

using Plots.PlotMeasures
gr() 
# Intial Conditions u0 and constants p0
const S0 = 1.0
u0 = [S0 * 1, 0.0, 0.0]
p0 = Float64[
    10.0, # τSI
    28, # τIR
    8/3 # τID
]
 
# Range for training Data
tspan = (0.0, 10)
datasize = 10
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
    Lux.Dense(25, 25, activation),
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
lossesRelu = Float32[]
lossesTanh = Float32[]
lossesSigmoid = Float32[]

# Call back to print iteration
callback5 = function (p, l, pred; doplot=true)
    if (activation == relu)
        push!(lossesRelu, l)
        if length(lossesRelu) % 50 == 0
            println("Current loss for relu after $(length(lossesRelu)) iterations: $(lossesRelu[end])")
        end
    elseif (activation == tanh)
        push!(lossesTanh, l)
        if length(lossesTanh) % 50 == 0
            println("Current loss for tanh after $(length(lossesTanh)) iterations: $(lossesTanh[end])")
        end
    else
        push!(lossesSigmoid, l)
        if length(lossesSigmoid) % 50 == 0
            println("Current loss for sigmoid after $(length(lossesSigmoid)) iterations: $(lossesSigmoid[end])")
        end
    end

    # # plot current prediction against data
    # if doplot && length(losses) % 1000 == 0
    #     plt = scatter(t, ode_data[1, :], label="data")
    #     scatter!(plt, t, pred[1, :], label="prediction")
    #     display(plot(plt))
    # end
    return false
end


pinit = ComponentArray(p)
callback5(pinit, loss_neuralode(pinit)...; doplot=true)

# Using Optimization.jl to solve the Neural ODE problem
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

# Using Adam Optimizer
result_neuralode = Optimization.solve(optprob, ADAM(0.01), callback=callback5, maxiters=20000)


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


plot(xtickfont=font("Times New Roman", 16),
ytickfont=font("Times New Roman", 16),
guidefont=font("Times New Roman", 18))

plot(title="Losses Over Time", xlabel="Time", ylabel="Loss", ylims=(0, 8000), xlims=(0, 10), size=plot_size, left_margin=left_margin, right_margin=right_margin, bottom_margin=bottom_margin, top_margin=top_margin)
# Add the data series
plot!(t, lossesRelu, lw=2, legend=false, seriestype=:line, label="relu", color=:red)  # Uncomment and provide data if needed
plot!(t, lossesTanh, lw=2, legend=false, seriestype=:line,label="tanh", color=:blue)
plot!(t, lossesSigmoid, lw=2, legend=false, seriestype=:line, label="sigmoid", color=:green)
plot!(legend=:outertopright, grid=true, legendfontsize=14)


# # Plot true soltion against Prediction
# plot!(ode_data', alpha=0.3, legend=false, label="True ODE Data")
# plot!(data_pred', label="Prediction")

# Plot u1
plot(tsteps, ode_data[1, :], seriestype=:scatter, marker=:circle, label="True u1", color=:red, alpha=0.3, xlabel="Time", ylabel="u", title="Lorenz System: Neural ODE")
plot!(tsteps, data_pred[1, :], seriestype=:line, label="Predicted u1", color=:red)

# Plot u2
plot!(tsteps, ode_data[2, :], seriestype=:scatter, marker=:circle, label="True u2", color=:green, alpha=0.3)
plot!(tsteps, data_pred[2, :], seriestype=:line, label="Predicted u2", color=:green)

# Plot u3
plot!(tsteps, ode_data[3, :], seriestype=:scatter, marker=:circle, label="True u3", color=:blue, alpha=0.3)
plot!(tsteps, data_pred[3, :], seriestype=:line, label="Predicted u3", color=:blue)

# Add legend and grid
plot!(legend=:topleft, grid=true)


# Extend the timespan for prediction
extended_tspan = (0.0, 15.0)
extended_datasize = 10
extended_tsteps = range(extended_tspan[1], extended_tspan[2], length=extended_datasize)

# Define the extended ODE problem with the trained parameters
prob_neuralode_extended = NeuralODE(dudt2, extended_tspan, Tsit5(), saveat=extended_tsteps)

# Use the trained parameters to make a prediction over the extended timespan
extended_prediction = predict_neuralode(result_neuralode2.u)

# Define the extended ODE problem for the true Lorenz system
extended_ode_prob = ODEProblem(Chaos!, u0, extended_tspan, p0)

# Solve the true Lorenz system over the extended timespan
extended_solution = Array(solve(extended_ode_prob, Tsit5(), saveat=extended_tsteps))

# Plot the true ODE data and the extended prediction
plot(extended_tsteps, extended_solution[1, :], seriestype=:scatter, marker=:circle, label="True u1", color=:red, xlabel="Time", ylabel="u", title="Lorenz System: Neural ODE Forecast")
plot!(extended_tsteps, extended_prediction[1, :], seriestype=:line, label="Predicted u1", color=:red)

plot!(extended_tsteps, extended_solution[2, :], seriestype=:scatter, marker=:circle, label="True u2", color=:green)
plot!(extended_tsteps, extended_prediction[2, :], seriestype=:line, label="Predicted u2", color=:green)

plot!(extended_tsteps, extended_solution[3, :], seriestype=:scatter, marker=:circle, label="True u3", color=:blue)
plot!(extended_tsteps, extended_prediction[3, :], seriestype=:line, label="Predicted u3", color=:blue)

# Add legend and grid
plot!(legend=:topleft, grid=true)