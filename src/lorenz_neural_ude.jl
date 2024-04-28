include("lorenz_ode.jl")
using .LorenzODE


module LorenzNeuralUDE

export Intialize, Run

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots


function InitializeNN(variables, weights, activation)
    rng = Random.default_rng()
    U = Lux.Chain(Lux.Dense(variables, weights, activation), Lux.Dense(weights, weights, activation), Lux.Dense(weights, variables))
    p, st = Lux.setup(rng, U)
    return p, st, U
end

function lorenzUde!(du, U, u, p, t, p_true)
    û = U(u, p, st)[1]
    LorenzODE.lorenz!(du, u, p_true, t)
    du .+= û
end

# Function to predict using the neural network
function predict(θ, prob_nn, X=u0, T=tspan)
    _prob = remake(prob_nn, p=θ)
    Array(solve(_prob, Vern7(), saveat=T))
end

# Loss function
function loss(θ, prob_nn, tspan)
    X̂ = predict(θ, prob_nn)
    X_true = Array(solve(prob, Vern7(), saveat=tspan))
    mean(abs2, X_true .- X̂)
end

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

function Run(U, u0, p, p_true, tspan, optimization, maxiters)
    nnDyanmics!(du, u, p, t) = lorenzUde!(du, U, u, p, t, p_true)
    prob_nn = ODEProblem(nnDyanmics!, u0, tspan, p_true)

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x, prob_nn, tspan), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

    result = Optimization.solve(optprob, optimization, callback=callback, maxiters=maxiters)
    return result
end


end