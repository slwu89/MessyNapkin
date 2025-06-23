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

const ad_sys = AutoForwardDiff()

# load the urchin data
urchin = CSV.read("./Urchin/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

# subset for speed
urchin = urchin[1:3:nrow(urchin), :]

# index into a single vector for b (random effects)
log_g_ix = 1:nrow(urchin)
log_p_ix = range(start=nrow(urchin)+1, length=nrow(urchin))

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
Log density (likelihood) of y and b hat, - \\log{f(y,b)}
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

# we'll need to optimize this with Newton's method to get b_hat, so need grad/Hess wrt b
function nlfyb_urchin(b, θ, urchin)
    -lfyb_urchin(b, θ, urchin)
end

prep_g_nlfyb = prepare_gradient(nlfyb_urchin, ad_sys, zero(b_init), Constant(th_init), Constant(urchin))

sp_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=Symbolics.SymbolicsSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
prep_h_nlfyb = prepare_hessian(nlfyb_urchin, sp_ad_sys, zero(b_init), Constant(th_init), Constant(urchin))

"""
Return s \\log{f_{\theta}(y,b)} + \\log{f_{\theta^{'}}(y,b)}
"""
function lfybs_urchin(b, s, θ, θ′, urchin)
    return s * lfyb_urchin(b, θ, urchin) + lfyb_urchin(b, θ′, urchin)
end

function nlfybs_urchin(b, s, θ, θ′, urchin)
    return -lfybs_urchin(b, s, θ, θ′, urchin)
end

# need to optimize with Newton's method, so need grad/Hess wrt b
prep_g_nlfybs = prepare_gradient(nlfybs_urchin, ad_sys, zero(b_init), Constant(0.0), Constant(th_init), Constant(th_init), Constant(urchin))
prep_h_nlfybs = prepare_hessian(nlfybs_urchin, sp_ad_sys, zero(b_init), Constant(0.0), Constant(th_init), Constant(th_init), Constant(urchin))

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

# for Q function, need to compute the derivative of the logdet hessian at s=0
function laplace_jl(s, b0, θ, θ′, urchin)

    g_nlfybs!(G, b) = gradient!(nlfybs_urchin, G, prep_g_nlfybs, ad_sys, b, Constant(s), Constant(θ), Constant(θ′), Constant(urchin))
    H_nlfybs!(H, b) = hessian!(nlfybs_urchin, H, prep_h_nlfybs, sp_ad_sys, b, Constant(s), Constant(θ), Constant(θ′), Constant(urchin))

    la_optim = optimize(
        b -> nlfybs_urchin(b, s, θ, θ′, urchin), 
        g_nlfybs!,
        H_nlfybs!,
        b0, 
        Newton(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
    )

    b_hat = Optim.minimizer(la_optim)
    H = hessian(nlfybs_urchin, prep_h_nlfybs, sp_ad_sys, b_hat, Constant(s), Constant(θ), Constant(θ′), Constant(urchin))
    R = pdR(H, 10)
    return logdet(R)
end

# finite diff for laplace_jl
fd_sys = AutoFiniteDiff()
laplace_jl_prep = prepare_derivative(laplace_jl, fd_sys, 0.0, Constant(b_init), Constant(th_init), Constant(th_init), Constant(urchin))

# the Q function
function Q(θ, θ′, urchin)
    # find b.hat maximising log joint density at θ′
    g_nlfyb!(G, b) = gradient!(nlfyb_urchin, G, prep_g_nlfyb, ad_sys, b, Constant(θ′), Constant(urchin))
    H_nlfyb!(H, b) = hessian!(nlfyb_urchin, H, prep_h_nlfyb, sp_ad_sys, b, Constant(θ′), Constant(urchin))

    la_optim = optimize(
         b -> nlfyb_urchin(b, θ′, urchin), 
         g_nlfyb!,
         H_nlfyb!,
         b_cache, 
         Newton(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
    )

    b_hat = Optim.minimizer(la_optim)
    b_cache .= b_hat # updated cached value for next iteration
    g = lfyb_urchin(b_hat, θ, urchin)

    # For s = -eps and eps find b maximising s log joint ## at th + log joint at thp along with log|H_s|.
    ds_H = derivative(laplace_jl, laplace_jl_prep, fd_sys, 0.0, Constant(b_hat), Constant(θ), Constant(θ′), Constant(urchin))

    return -(g - ds_H)
end

θ = zero(th_init)
θ′ = zero(th_init)

# b_cache .= b_init
const b_cache = deepcopy(b_init)
er_min = 0.0

for i in 1:30
    er = optimize(
        θ -> Q(θ, θ′, urchin),
        θ,
        NelderMead(;
            parameters = Optim.FixedParameters()
            # parameters = Optim.FixedParameters(α = 1.0, β = 0.5, γ = 2.0, δ = 0.5)
        ),
        Optim.Options(show_trace=true, iterations=100)
    )
    er_min = Optim.minimum(er)
    θ = deepcopy(Optim.minimizer(er))
    θ′ = deepcopy(Optim.minimizer(er))
    println("iteration: $i, θ: $θ")
end

θ
er_min