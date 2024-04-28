include("lorenz_ode.jl")
include("lorenz_nueral_ode.jl")

using .LorenzODE
using .LorenzNueralODE

using Plots, Optimization, OptimizationOptimJL, OptimizationOptimisers



u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 100.0)
datasize = 100
tsteps = range(tspan[1], tspan[2]; length=datasize)

####  ODE Solution ####

# solve the ODE for Lorenz attractor eqautions
sol = LorenzODE.Lorenz(u0, tspan, tsteps)
trueOdeData = Array(sol)
# Parameter knobs to contorl accuracy of the model
equations = 3
weights = 100
activation = tanh

#### Neural ODE Solution ####

# Intialize Neural Network  with 3 layers
p_ode, st_ode, lorenzNNOde = LorenzNueralODE.InitializeNN(equations, weights, activation, tspan, tsteps)

# Run the neural network with  defined parameters
optimizer = Optimisers.Adam(0.001)
maxiters = 500

result_ode = LorenzNueralODE.Run(trueOdeData, u0, p_ode, st_ode, lorenzNNOde, optimizer, maxiters)


#### Neural UDE Solution ####