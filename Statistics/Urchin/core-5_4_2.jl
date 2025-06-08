# urchins model: random effects, MLE by EM algorithm
using DataFrames, CSV
using Distributions
using LinearAlgebra

using Optim
using LineSearches
using Plots
using DifferentiationInterface
import ForwardDiff

# for sparse AD
using SparseMatrixColorings
import Symbolics

const ad_sys = AutoForwardDiff()

# load the urchin data
const urchin = CSV.read("./Urchin/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

# index into a single vector for b (random effects)
const log_g_ix = 1:nrow(urchin)
const log_p_ix = range(start=nrow(urchin)+1, length=nrow(urchin))

# index into a single vector for θ (fixed effects)
const log_ω_ix = 1
const μ_g_ix = 2
const log_σ_g_ix = 3
const μ_p_ix = 4
const log_σ_p_ix = 5
const log_σ_ix = 6

"""
The biological model for a single urchin; `g` and `p`
    are the individual level random effects
"""
function model_urchin_vol(ω, g, p, a)
    aₘ = log(p / (g*ω))/g
    return ifelse(a < aₘ, ω*exp(g*a), p/g + p*(a-aₘ))
end

# lyfb in the R code, but need to do the grad/hessian seperately
"""
Log density (likelihood) of y and b, \\log{f_{\theta}(y,b)}
"""
function lfyb_urchin(b, θ, urchin)
    # extract fixed effects
    σ = exp(θ[log_σ_ix])
    σ_g = exp(θ[log_σ_g_ix])
    σ_p = exp(θ[log_σ_p_ix])
    ω = exp(θ[log_ω_ix])
    # extract random effects
    log_g = b[log_g_ix]
    log_p = b[log_p_ix]
    # calculate the loglikelihood of y and b
    logll = 0.0
    for i in axes(urchin,1)
        v = model_urchin_vol(ω, exp(log_g[i]), exp(log_p[i]), urchin[i, :age])
        logll += logpdf(Normal(sqrt(v), σ), sqrt(urchin[i, :vol]))
        logll += logpdf(Normal(θ[μ_g_ix], σ_g), log_g[i])
        logll += logpdf(Normal(θ[μ_p_ix], σ_p), log_p[i])
    end
    return logll
end

"""
Return s \\log{f_{\theta}(y,b)} + \\log{f_{\theta^{'}}(y,b)}
"""
function lyfbs_urchin(b, s, θ, θ′, urchin)
    return s * lfyb_urchin(b, θ, urchin) + lfyb_urchin(b, θ′, urchin)
end

# initial values for fixed/random effects
th = [
    -3.0,
    -0.3, 
    -1.5,
    0.15,
    -1.5,
    -1.37
]
n = nrow(urchin)
b = [fill(th[2],n); fill(th[4],n)]

# grad/hess for lyfb
prep_g_lfyb = prepare_gradient(lfyb_urchin, ad_sys, zero(b), Constant(th), Constant(urchin))

sp_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=Symbolics.SymbolicsSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
prep_sp_h_lfyb = prepare_hessian(lfyb_urchin, sp_ad_sys, rand(length(b)), Constant(th), Constant(urchin))

lfyb_urchin(b, th, urchin)
gradient(lfyb_urchin, prep_g_lfyb, ad_sys, b, Constant(th), Constant(urchin))
hessian(lfyb_urchin, prep_sp_h_lfyb, sp_ad_sys, b, Constant(th), Constant(urchin))    

# grad/hess for lyfbs wrt b
prep_g_lfybs = prepare_gradient(lyfbs_urchin, ad_sys, zero(b), Constant(0.0), Constant(th), Constant(th), Constant(urchin))
prep_sp_h_lfybs = prepare_hessian(lyfbs_urchin, sp_ad_sys, rand(length(b)), Constant(0.0), Constant(th), Constant(th), Constant(urchin))

lyfbs_urchin(b, 0.1, th, th, urchin)
gradient(lyfbs_urchin, prep_g_lfybs, ad_sys, b, Constant(0.1), Constant(th), Constant(th), Constant(urchin))
hessian(lyfbs_urchin, prep_sp_h_lfybs, sp_ad_sys, b, Constant(0.1), Constant(th), Constant(th), Constant(urchin))    

# laplace for s=0
# this function should find \hat{b} for lfyb_urchin using θ′ but then
# return or somehow cache \hat{b} and return lfyb_urchin evaluated at \hat{b} and θ (not prime)
function laplace(θ, θ′, urchin)
end

# laplce for s!=0
# find \hat{b} for lfybs_urchin return the logdet(H)/2 of H evaluated at \hat{b} and s=0
function laplace(θ, θ′, urchin)
end

# Q function