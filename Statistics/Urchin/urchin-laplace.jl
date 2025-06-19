# compare the basic likelihood model
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
using SparseArrays

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

# initial starting values for Θ
th_init = [
    -4.0,
    -0.2, 
    log(0.1),
    0.2,
    log(0.1),
    log(0.5)
]

# initial starting values for b
b_init = [
    fill(th_init[μ_g_ix], nrow(urchin)); fill(th_init[μ_p_ix], nrow(urchin))
]

"""
The biological model for a single urchin; `g` and `p`
    are the individual level random effects
"""
function model_urchin_vol(ω, g, p, a)
    aₘ = log(p / (g*ω))/g    
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

const ad_sys = AutoForwardDiff()
const prep_g_nlyfb = prepare_gradient(nlyfb_urchin, ad_sys, zero(b_init), Constant(th_init), Constant(urchin))

const sp_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=Symbolics.SymbolicsSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
const prep_sp_h_nlyfb = prepare_hessian(nlyfb_urchin, sp_ad_sys, rand(length(b_init)), Constant(th_init), Constant(urchin))

# pdR in Julia
function cholesky_check(m)
     try
          # cholesky(m).U # only for dense
          sparse(cholesky(m))
     catch
         nothing
     end
end

function cholesky_check(m::T) where {T<:SparseMatrixCSC}
     try
          # thanks to https://discourse.julialang.org/t/how-to-get-upper-triangular-cholesky-factor-of-sparse-matrix/129979/2?u=slwu89
          C = cholesky(m)
          perm = C.p
          invperm_vec = invperm(perm)
          L = sparse(C.L)[invperm_vec, invperm_vec]
          L'
     catch
         nothing
     end
end
 
function pdR(H, k_mult=20, tol=eps()^.8)
     k=1
     tol = tol * opnorm(H, 1)
     n = size(H, 2)
     R = cholesky_check(H + (k-1)*tol*I(n))
     while isnothing(R)
         k *= k_mult
         R = cholesky_check(H + (k-1)*tol*I(n))
     end
     return R
end

"""
Store the value of \\hat{b} between calls to `marg_nll` for better starting points
for inner optimization of b
"""
const b_cache = deepcopy(b_init)

"""
The marginal log-likelihood of the fixed effects L(Θ) = \\int f_{\\Theta}(y,b) db
"""
function marginal_nll(θ)
     # the `llu` function from Wood's R code
     nb = length(b_cache)

     g_nlyfb!(G, b) = gradient!(nlyfb_urchin, G, prep_g_nlyfb, ad_sys, b, Constant(θ), Constant(urchin))
     H_nlyfb!(H, b) = hessian!(nlyfb_urchin, H, prep_sp_h_nlyfb, sp_ad_sys, b, Constant(θ), Constant(urchin))

     nlfyb_optim = optimize(
          b -> nlyfb_urchin(b, θ, urchin), 
          g_nlyfb!,
          H_nlyfb!,
          b_cache, 
          Newton(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
     )
     b_hat = Optim.minimizer(nlfyb_optim)
     nlfyb_hat = Optim.minimum(nlfyb_optim)

     b_cache .= b_hat # updated cached value for next iteration

     H = hessian(nlyfb_urchin, prep_sp_h_nlyfb, sp_ad_sys, b_hat, Constant(θ), Constant(urchin))  
     R = pdR(H)
     
     return nlfyb_hat + logdet(R) - log(2π)*(nb/2)
end

# auto finite diff for the outer marginal nll
fd_sys = AutoFiniteDiff()
marginal_nll_prep = prepare_gradient(marginal_nll, fd_sys, rand(length(th_init)))
g_marginal_nll!(G, θ) = gradient!(marginal_nll, G, marginal_nll_prep, fd_sys, θ)

marginal_mle = optimize(
     marginal_nll,
     g_marginal_nll!,
     th_init,
     LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking()),
     Optim.Options(show_trace=true)
)

aic = 2*Optim.minimum(marginal_mle) + 2*length(th_init)
Optim.minimizer(marginal_mle)