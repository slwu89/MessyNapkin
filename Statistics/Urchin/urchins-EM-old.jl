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
urchin = CSV.read("./Urchin/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

# for testing
urchin = urchin[[1,25,50,75,100,115,125,142], :]

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
    ll = 0.0
    for i in axes(urchin,1)
        v = model_urchin_vol(ω, exp(log_g[i]), exp(log_p[i]), urchin[i, :age])
        ll += logpdf(Normal(sqrt(v), σ), sqrt(urchin[i, :vol]))
        ll += logpdf(Normal(θ[μ_g_ix], σ_g), log_g[i])
        ll += logpdf(Normal(θ[μ_p_ix], σ_p), log_p[i])
    end
    return ll
end

function nlfyb_urchin(b, θ, urchin)
    return -lfyb_urchin(b, θ, urchin)
end

"""
Return s \\log{f_{\theta}(y,b)} + \\log{f_{\theta^{'}}(y,b)}
"""
function lyfbs_urchin(b, s, θ, θ′, urchin)
    return s * lfyb_urchin(b, θ, urchin) + lfyb_urchin(b, θ′, urchin)
end

function nlyfbs_urchin(b, s, θ, θ′, urchin)
    return -lyfbs_urchin(b, s, θ, θ′, urchin)
end

"""
Find \\hat{b} optimizing s \\log{f_{\theta}(y,b)} + \\log{f_{\theta^{'}}(y,b)}
and then return the log determinant of its Hessian evaluated at \\hat{b}
"""
function laplace_urchin(b, s, θ, θ′, urchin)
    nlfybs_b_hat = optimize(
        b -> nlyfbs_urchin(b, s, θ, θ′, urchin), 
        g_nlfybs!,
        b, 
        LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
    )
    b_hat = Optim.minimizer(nlfybs_b_hat)
    H = hessian(lyfbs_urchin, prep_sp_h_lfybs, sp_ad_sys, b_hat, Constant(s), Constant(θ), Constant(θ′), Constant(urchin))
    return logdet(-H)
end

# grad for nlfyb_urchin (to find \hat{b})
prep_g_nlfyb = prepare_gradient(nlfyb_urchin, ad_sys, zero(b), Constant(th), Constant(urchin))
# gradient(nlfyb_urchin, prep_g_nlfyb, ad_sys, b, Constant(th), Constant(urchin))
g_nlfyb!(G, b) = gradient!(nlfyb_urchin, G, prep_g_nlfyb, ad_sys, b, Constant(th), Constant(urchin))

# grad for nlyfbs_urchin (to minimize it and find b)
prep_g_nlfybs = prepare_gradient(nlyfbs_urchin, ad_sys, zero(b), Constant(0.0), Constant(th), Constant(th), Constant(urchin))
# gradient(nlyfbs_urchin, prep_g_nlfybs, ad_sys, b, Constant(0.1), Constant(th), Constant(th), Constant(urchin))
g_nlfybs!(G, b) = gradient!(nlyfbs_urchin, G, prep_g_nlfybs, ad_sys, b, Constant(0.1), Constant(th), Constant(th), Constant(urchin))

# hess for lyfbs_urchin (to approximate Q)
sp_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=Symbolics.SymbolicsSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
prep_sp_h_lfybs = prepare_hessian(lyfbs_urchin, sp_ad_sys, rand(length(b)), Constant(0.1), Constant(th), Constant(th), Constant(urchin))
# hessian(lyfbs_urchin, prep_sp_h_lfybs, sp_ad_sys, b, Constant(0.1), Constant(th), Constant(th), Constant(urchin))

# Q function
function Q(θ, θ′, urchin)
    # optimize nlfyb_urchin wrt to θ′ to find \hat{b}
    # evaluate lfyb_urchin at \hat{b} and θ to get g
    nlfyb_b_hat = optimize(
        b -> nlfyb_urchin(b, θ′, urchin), 
        g_nlfyb!,
        b_cache, 
        LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
    )
    b_hat = Optim.minimizer(nlfyb_b_hat)
    g = lfyb_urchin(b_hat, θ, urchin)
    b_cache .= b_hat # update cache

    # optimize nlyfbs_urchin to find second \hat{b}
    # H = - hessian of lyfbs_urchin at that second \hat{b}
    # logdet(H)/2 is the element we need to combine with g

    # we need to do it like the code in the book. here is why
    #   * if we find b_hat at s=0, it is just like what we did previously
    #  need a simpler ver of `laplace` from the book which does the optim
    # to find b_hat and then returns the logdet(H) evaluated at b_hat

    eps = 1E-5
    lap = laplace_urchin(b_hat, eps/2, θ, θ′, urchin)
    lam = laplace_urchin(b_hat, -eps/2, θ, θ′, urchin)

    return -(g - (lap-lam)/eps)
end

# basic EM
const b_cache = deepcopy(b)
thp = copy(th);
for i in 1:30
    er = optimize(
        θ -> Q(θ, thp, urchin),
        th,
        NelderMead(),
        Optim.Options(show_trace=true, iterations=50)
    )
    th = Optim.minimizer(er)
    thp = Optim.minimizer(er)
    println("theta: $(th)")
end

# test within Q
thp = zeros(6)

la = optimize(
    b -> nlfyb_urchin(b, thp, urchin), 
    g_nlfyb!,
    b, 
    LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
)
b_hat = Optim.minimizer(la)
lfyb_urchin(b_hat, th, urchin)
lfyb_urchin(b, th, urchin)



eps = 1E-5
lap = laplace_urchin(b, eps/2, th, thp, urchin)
lam = laplace_urchin(b, -eps/2, th, thp, urchin)

function laplace_urchin_test(b, s, θ, θ′, urchin)
    nlfybs_b_hat = optimize(
        b -> nlyfbs_urchin(b, s, θ, θ′, urchin), 
        g_nlfybs!,
        b, 
        LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
    )
    b_hat = Optim.minimizer(nlfybs_b_hat)
    H = hessian(lyfbs_urchin, prep_sp_h_lfybs, sp_ad_sys, b_hat, Constant(s), Constant(θ), Constant(θ′), Constant(urchin))
    return H
end

lap_H = laplace_urchin_test(b, eps/2, th, thp, urchin)
cholesky(lap_H)

x=1
try
    x=sqrt(-2)
catch
end

using Tables
CSV.write("./Urchin/H.csv", Tables.table(Matrix(lap_H)), writeheader=false)

opnorm(lap_H, 1)

function pdR(H, k_mult=20, tol=2.220446e-16^.8)
    k=1
    tol = tol * opnorm(H, 1)
    n = size(H, 2)
    R = similar(H)
    while begin
        try
            R = chol(H + (k-1)*tol*I(n))
            false
        catch _
            true
        end
    end
        k *= k_mult
    end
    return R
end



pdR(lap_H, 10)

# pdR <- function(H,k.mult=20,tol=.Machine$double.eps^.8) {
#     k <- 1; tol <- tol * norm(H); n <- ncol(H)
#     while (inherits(try(R <- chol(H + (k-1)*tol*diag(n)),
#            silent=TRUE),"try-error")) k <- k * k.mult
#     R 
# }

m = [5 1
    1 3]
cholesky(m)