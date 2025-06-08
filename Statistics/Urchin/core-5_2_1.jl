# urchins model (no random effects)
using DataFrames, CSV
using Distributions
using LinearAlgebra

using Optim
using LineSearches
using Plots, StatsPlots
using DifferentiationInterface
import ForwardDiff

ad_sys = AutoForwardDiff()

const urchin = CSV.read("./MLE/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

"""
The biological model: returns vector of urchin volumes
"""
function model_urchin_vol(Θ, urchin)
    ω, γ, ϕ, σ = exp.(Θ)
    aₘ = log(ϕ / (γ*ω))/γ
    return [a < aₘ ? ω*exp(γ*a) : ϕ/γ + ϕ*(a-aₘ) for a in urchin.age]
end

Θ0 = Float64[0,0,0,0]

model_urchin_vol(Θ0,urchin)

"""
Negative log likelihood of urchin model
"""
function model_urchin_nll(Θ, urchin)
    σ = exp(Θ[4])
    V = model_urchin_vol(Θ, urchin)
    -sum([logpdf(Normal(sqrt(V[i]), σ), sqrt(urchin[i, :vol])) for i in axes(urchin, 1)])
end

# model_urchin_nll(Θ0,urchin)

# one argument closure; not the most efficient, see https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/dev/tutorials/advanced/#Contexts
model_urchin_f = Θ -> model_urchin_nll(Θ, urchin)

prep_grad_urchin = prepare_gradient(model_urchin_f, ad_sys, zero(Θ0))
grad_urchin!(G, Θ) = gradient!(model_urchin_f, G, prep_grad_urchin, ad_sys, Θ)

# default HagerZhang line search is too aggressive
result_urchin = optimize(model_urchin_f, grad_urchin!, Θ0, LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking()))


exp.(Optim.minimizer(result_urchin))
Optim.minimum(result_urchin)

mle_urchin = Optim.minimizer(result_urchin)
v = model_urchin_vol(mle_urchin, urchin)
rsd = @. (sqrt(urchin.vol) - sqrt(v)) / exp(mle_urchin[4])

scatter(v, rsd, legend=false)
qqnorm(rsd) 

V = inv(hessian(model_urchin_f, ad_sys, mle_urchin))
sd = sqrt.(diag(V))

exp.(mle_urchin - 2*sd)
exp.(mle_urchin + 2*sd)