using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, DifferentialEquations, Statistics
rng = Random.default_rng()


# Intial Conditions u0 and constants p0
const S0 = 1.0
u0 = [S0 * 1, 0.0, 0.0]
p0 = Float64[
    10.0, # τSI
    28, # τIR
    8/3 # τID
]

# Range for training Data
tspan = (0.0, 25)
datasize = 25
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
    Lux.Dense(3, 100, activation),
    Lux.Dense(100, 200, activation),
    Lux.Dense(200, 100, activation),
    Lux.Dense(100, 3))
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
result_neuralode = Optimization.solve(optprob, ADAM(0.01), callback=callback5, maxiters=6000)

optprob2 = remake(optprob, u0=result_neuralode.u)
result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.1),
    callback=callback5,
    allow_f_increases=false)
data_pred = predict_neuralode(result_neuralode2.u)


# Plot true soltion against Prediction
plot!(ode_data', alpha=0.3, legend=false, label="True ODE Data")
plot!(data_pred', label="Prediction")

