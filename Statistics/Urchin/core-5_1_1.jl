# examples from section 5.1.1 Newton's Method
# in Wood's Core Statistics

# look at this to potentially provide a speed up
# https://julianlsolvers.github.io/Optim.jl/stable/user/tipsandtricks/#Avoid-repeating-computations

using DataFrames
using Distributions
using Optim
using LineSearches
using Plots
using DifferentiationInterface
import ForwardDiff

# use ForwardDiff because it's simple
ad_sys = AutoForwardDiff()

# ------------------------------------------------------------
# the cells example
const cell = DataFrame(t=2:14, y=[35,33,33,39,24,25,18,20,23,13,14,20,18])

function model_cell(delta, cell)
    # expected values of counts
    mu = @. 50*exp(-delta*cell.t)
    # neg log likelihood
    -sum([logpdf(Poisson(mu[i]), cell[i, :y]) for i in axes(cell,1)])
end

# a starting value
delta_0 = [5.0]

# one argument closure; not the most efficient, see https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/dev/tutorials/advanced/#Contexts
model_cell_f = x -> model_cell(x, cell)

prep_grad_cell = prepare_gradient(model_cell_f, ad_sys, zero(delta_0))
grad_cell!(G, x) = gradient!(model_cell_f, G, prep_grad_cell, ad_sys, x)

prep_hess_cell = prepare_hessian(model_cell_f, ad_sys, zero(delta_0))
hess_cell!(H, x) = hessian!(model_cell_f, H, prep_hess_cell, ad_sys, x)

# default HagerZhang line search is too aggressive
result_cell = optimize(model_cell_f, grad_cell!, hess_cell!, delta_0, Newton(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking()))
Optim.minimizer(result_cell)

# ------------------------------------------------------------
# the AIDS example
# model parameterized with years *since* 1980
const aids = DataFrame(t=(81:93) .- 80, cases=[12,14,33,50,67,74,123,141,165,204,253,246,240])

# typically in Julia most of the optimization packages need the parameters as a single vector
function model_aids(pars, aids)
    alpha, beta = exp.(pars)
    # expected values of cases
    mu = @. alpha*exp(beta*aids.t)
    # neg log likelihood
    -sum([logpdf(Poisson(mu[i]), aids[i, :cases]) for i in axes(aids,1)])
end

# pars should be positive; optimize on log scale
pars_0 = log.([4,0.35])

model_aids_f = x -> model_aids(x, aids)

prep_grad_aids = prepare_gradient(model_aids_f, ad_sys, zero(pars_0))
grad_aids!(G, x) = gradient!(model_aids_f, G, prep_grad_aids, ad_sys, x)

prep_hess_aids = prepare_hessian(model_aids_f, ad_sys, zero(pars_0))
hess_aids!(H, x) = hessian!(model_aids_f, H, prep_hess_aids, ad_sys, x)

# default HagerZhang line search is too aggressive
result_aids = optimize(model_aids_f, grad_aids!, hess_aids!, pars_0, Newton(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking()))

min_aids = exp.(Optim.minimizer(result_aids))
cov_aids = inv(hessian(model_aids_f, prep_hess_aids, ad_sys, min_aids))

scatter(aids.t, aids.cases, legend=false)
plot!(aids.t, @. min_aids[1]*exp(min_aids[2]*aids.t))
