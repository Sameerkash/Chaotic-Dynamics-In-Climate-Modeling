using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, DifferentialEquations, Statistics
using Plots.PlotMeasures
    rng = Random.default_rng()
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

# Add noise to the ODE data
noise_level = 0.1
noisy_ode_data = ode_data .+ noise_level * randn(size(ode_data))

# Definition of Neural Network with activation function and layers.
activation = sigmoid
dudt2 = Lux.Chain(
    Lux.Dense(3, 100, activation),
    Lux.Dense(100, 3))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=t)

# Prediction function called in iterations
function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

# Function defined for minimizing cost/loss of the solution
function loss_neuralode(p, data)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss, pred
end

# Training with original data
losses_original = Float32[]
callback_original = function (p, l, pred; doplot=true)
    push!(losses_original, l)
    if length(losses_original) % 50 == 0
        println("Current loss after $(length(losses_original)) iterations: $(losses_original[end])")
    end
    return false
end

pinit = ComponentArray(p)
callback_original(pinit, loss_neuralode(pinit, ode_data)...; doplot=true)
adtype = Optimization.AutoZygote()
optf_original = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x, ode_data), adtype)
optprob_original = Optimization.OptimizationProblem(optf_original, pinit)
result_neuralode_original = Optimization.solve(optprob_original, ADAM(0.01), callback=callback_original, maxiters=50000)
optprob2_original = remake(optprob_original, u0=result_neuralode_original.u)
result_neuralode2_original = Optimization.solve(optprob2_original, Optim.BFGS(initial_stepnorm=0.1), callback=callback_original, allow_f_increases=false)
data_pred_original = predict_neuralode(result_neuralode2_original.u)

# Training with noisy data
losses_noisy = Float32[]
callback_noisy = function (p, l, pred; doplot=true)
    push!(losses_noisy, l)
    if length(losses_noisy) % 50 == 0
        println("Current loss after $(length(losses_noisy)) iterations: $(losses_noisy[end])")
    end
    return false
end

callback_noisy(pinit, loss_neuralode(pinit, noisy_ode_data)...; doplot=true)
optf_noisy = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x, noisy_ode_data), adtype)
optprob_noisy = Optimization.OptimizationProblem(optf_noisy, pinit)
result_neuralode_noisy = Optimization.solve(optprob_noisy, ADAM(0.01), callback=callback_noisy, maxiters=50000)
optprob2_noisy = remake(optprob_noisy, u0=result_neuralode_noisy.u)
result_neuralode2_noisy = Optimization.solve(optprob2_noisy, Optim.BFGS(initial_stepnorm=0.1), callback=callback_noisy, allow_f_increases=false)
data_pred_noisy = predict_neuralode(result_neuralode2_noisy.u)


plot_size = (1200, 600)# Width x Height in pixels
left_margin = 15px
right_margin = 15px
top_margin = 15px
bottom_margin = 15px
# Plot losses
plot(t, losses_original, title="Losses Over Time (Original vs Noisy)", xlabel="Time", ylabel="Loss", lw=1, legend=false, seriestype=:line, label="Original", )
plot!(t, losses_noisy, lw=1, seriestype=:line, label="Noisy")

# Plot u1
plot(t, ode_data[1, :], seriestype=:scatter, marker=:circle, label="True u1 (Original)", color=:red, alpha=0.3, xlabel="Time", ylabel="u", title="Lorenz System: Neural ODE (Original vs Noisy)", size=plot_size, left_margin=left_margin, right_margin=right_margin,  bottom_margin=bottom_margin, top_margin=top_margin)
plot!(t, data_pred_original[1, :], seriestype=:line, label="Predicted u1 (Original)", color=:red)
plot!(t, noisy_ode_data[1, :], seriestype=:scatter, marker=:x, label="True u1 (Noisy)", color=:blue, alpha=0.3)
plot!(t, data_pred_noisy[1, :], seriestype=:line, label="Predicted u1 (Noisy)", color=:blue)

# Plot u2
plot!(t, ode_data[2, :], seriestype=:scatter, marker=:circle, label="True u2 (Original)", color=:green, alpha=0.3)
plot!(t, data_pred_original[2, :], seriestype=:line, label="Predicted u2 (Original)", color=:green)
plot!(t, noisy_ode_data[2, :], seriestype=:scatter, marker=:x, label="True u2 (Noisy)", color=:purple, alpha=0.3)
plot!(t, data_pred_noisy[2, :], seriestype=:line, label="Predicted u2 (Noisy)", color=:purple)

# Plot u3
plot!(t, ode_data[3, :], seriestype=:scatter, marker=:circle, label="True u3 (Original)", color=:orange, alpha=0.3)
plot!(t, data_pred_original[3, :], seriestype=:line, label="Predicted u3 (Original)", color=:orange)
plot!(t, noisy_ode_data[3, :], seriestype=:scatter, marker=:x, label="True u3 (Noisy)", color=:brown, alpha=0.3)
plot!(t, data_pred_noisy[3, :], seriestype=:line, label="Predicted u3 (Noisy)", color=:brown)



# Add legend and grid
plot!(legend=:outertopright, grid=true)
