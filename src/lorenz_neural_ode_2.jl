using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, DifferentialEquations, Statistics

### definiton of Lorenz system of ordinary differentail equations ###
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end


### co-efficients and inital conditions ###
u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 100.0)
datasize = 100


σ = 10.0
ρ = 28.0
β = 8 / 3
p = [σ, ρ, β]

prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob)

plot1 = plot!(
    sol,
    vars=(1, 2, 3),
    xlabel="x",
    ylabel="y",
    zlabel="z",
    label="x against y against z"
)


tsteps = sol.t
trueOdeData = Array(sol)
equations = 3
activation = relu
optimizer = Optimisers.Adam()
maxiters = 1000


rng = Random.default_rng()

lorenzNN = Lux.Chain(Lux.Dense(equations, 50, activation),
    Lux.Dense(50, 50, activation),
    Lux.Dense(50, equations)
)
p, st = Lux.setup(rng, lorenzNN)

lorenzNNOde = NeuralODE(lorenzNN, tspan, Tsit5(), saveat=tsteps)

function prediction(p)
    Array(lorenzNNOde(u0, p, st)[1])
end

function loss(p)
    pred = prediction(p)
    loss = sum(abs2, trueOdeData .- pred)
    return loss, pred
end

losses = Float64[]

callback = function (p, l, doplot=true)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end

    return false
end

pinit = ComponentArray(p)
adtype = Optimization.AutoZygote()

optimizeFunction = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
neuralProblem = Optimization.OptimizationProblem(optimizeFunction, pinit)

result = Optimization.solve(neuralProblem, optimizer; callback=callback, maxiters=maxiters)

p_trained = result.u

ts = sol.t
X̂ = prediction(p_trained)


plot2 = plot(
    ts,
    transpose(X̂),
    # vars=(1, 2, 3),
    xlabel="x",
    ylabel="y",
    zlabel="z",
    label="x against y against z"
)
