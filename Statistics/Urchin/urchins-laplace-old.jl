# urchins model: random effects, MLE by direct Laplace approximation
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
    # return a < aₘ ? ω*exp(g*a) : p/g + p*(a-aₘ)
    return ifelse(a < aₘ, ω*exp(g*a), p/g + p*(a-aₘ))
end

"""
Negative log density (likelihood) of y and b hat, - \\log{f(y,b)}
"""
function nlyfb_urchin(b, θ, urchin)
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
    return -logll
end

# initial starting values for Θ
θ_init = [
    -4.0,
    -0.2, 
    log(0.1),
    0.2,
    log(0.1),
    log(0.5)
]

# initial starting values for b
b_init = [
    fill(θ_init[μ_g_ix], nrow(urchin)); fill(θ_init[μ_p_ix], nrow(urchin))
]

const prep_g_nlyfb = prepare_gradient(nlyfb_urchin, ad_sys, zero(b_init), Constant(θ_init), Constant(urchin))
g_nlyfb!(G, b) = gradient!(nlyfb_urchin, G, prep_g_nlyfb, ad_sys, b, Constant(θ_init), Constant(urchin))


# --------------------------------------------------
# Hessian prep

const sp_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=Symbolics.SymbolicsSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
const prep_sp_h_nlyfb = prepare_hessian(nlyfb_urchin, sp_ad_sys, rand(length(b_init)), Constant(θ_init), Constant(urchin))


# ------------------------------------------------------------
# marginal log density of Theta L(Theta) = \int f_{\Theta}(y,b) db

"""
Store the value of \\hat{b} between calls to `marg_nll` for better starting points
for inner optimization of b
"""
const b_cache = deepcopy(b_init)

"""
The marginal log-likelihood of the fixed effects L(Θ) = \\int f_{\\Theta}(y,b) db
"""
function marg_nll(Θ)
    # the `llu` function from Wood's R code
    nb = length(b_cache)

    f_yb_mle = optimize(
        b -> nlyfb_urchin(b, Θ, urchin), 
        g_nlyfb!,
        b_cache, 
        LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
    )
    b_hat = Optim.minimizer(f_yb_mle)
    b_cache .= b_hat # updated cached value for next iteration
    f_yb = Optim.minimum(f_yb_mle)

    H = hessian(nlyfb_urchin, prep_sp_h_nlyfb, sp_ad_sys, b_hat, Constant(Θ), Constant(urchin))    
    return f_yb - 0.5 * (log((2π)^nb) - logdet(H))
end

# marg_nll(θ_init)

fd_sys = AutoFiniteDiff()
const marg_nll_prep = prepare_gradient(marg_nll, fd_sys, rand(length(θ_init)))
g_marg_nll!(G, θ) = gradient!(marg_nll, G, marg_nll_prep, fd_sys, θ)

marginal_mle = optimize(
    marg_nll,
    g_marg_nll!,
    θ_init,
    LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking()),
    Optim.Options(show_trace=true)
)
aic = 2*Optim.minimum(marginal_mle) + 2*length(θ_init)
θ_mle = Optim.minimizer(marginal_mle)