

module LorenzNueralODE

export InitializeNN, Run

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots

function InitializeNN(variables, weights, activation, tspan, tsteps)
    rng = Random.default_rng()
    lorenzNN = Lux.Chain(Lux.Dense(variables, weights, activation), Lux.Dense(weights, weights, activation), Lux.Dense(weights, variables))
    p, st = Lux.setup(rng, lorenzNN)
    lorenzNNOde = NeuralODE(lorenzNN, tspan, Tsit5(), saveat=tsteps)
    return p, st, lorenzNNOde
end

function prediction(u0, p, st, lorenzNNOde)
    Array(lorenzNNOde(u0, p, st)[1])
end

function loss(trueOdeData, u0, p, st, lorenzNNOde)
    pred = prediction(u0, p, st, lorenzNNOde)
    loss = sum(abs2, trueOdeData .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot=true)
    println(l)
    return false
end

function Run(trueOdeData, u0, p, st, lorenzNNOde, optimizer, maxiters)
    pinit = ComponentArray(p)
    adtype = Optimization.AutoZygote()
    optimizeFunction = Optimization.OptimizationFunction((x, p) -> loss(trueOdeData, u0, x, st, lorenzNNOde), adtype)
    neuralProblem = Optimization.OptimizationProblem(optimizeFunction, pinit)

    result = Optimization.solve(neuralProblem, optimizer; callback=callback, maxiters=maxiters)
    return result
end

end