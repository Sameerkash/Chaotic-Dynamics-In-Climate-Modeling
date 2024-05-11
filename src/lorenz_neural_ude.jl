module LorenzNeuralUDE

export Intialize, Run

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, StableRNGs


function InitializeNN(variables, weights, activation)
    rng = StableRNG(1111)

    U = Lux.Chain(Lux.Dense(variables, weights, activation), Lux.Dense(weights, weights, activation), Lux.Dense(weights, variables))
    p, st = Lux.setup(rng, U)
    return p, st, U
end

function lorenzUde!(du, u, p, t, p_true, st, U)
    û = U(u, p, st)[1]
    du[1] = p_true[1] * (u[2] - û[1])
    du[2] = -u[2] + û[2]
    du[3] = -p_true[3] * u[3] + û[3]
end

# Function to predict using the neural network
function predict(θ, prob_nn, X, T=tspan)
    _prob = remake(prob_nn, u0=X, tspan=(T[1], T[end]), p=θ)
    Array(solve(_prob, Vern7(), saveat=T, abstol = 1e-6, reltol = 1e-6,))
end

# Loss function
function loss(θ, prob_nn, tspan, X)
    X̂ = predict(θ, prob_nn, X, tspan)
    mean(abs2, X .- X̂)
end

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

function Run(U, p, p_true, st_ude, tspan, X, optimization, maxiters)
    nnDyanmics!(du, u, p, t) = lorenzUde!(du, u, p, t, p_true, st_ude,U)
    prob_nn = ODEProblem(nnDyanmics!, X[:, 1], tspan, p)

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x, prob_nn, tspan, X), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

    result = Optimization.solve(optprob, optimization, callback=callback, maxiters=maxiters)
    return result
end


end

