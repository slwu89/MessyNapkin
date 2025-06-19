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

using RCall

# load the urchin data
urchin = CSV.read("./Urchin/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

urchin_subset = urchin[[1,3,5,10,50,100,142], :]

# index into a single vector for b (random effects)
log_g_ix = 1:nrow(urchin_subset)
log_p_ix = range(start=nrow(urchin_subset)+1, length=nrow(urchin_subset))

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
    fill(th_init[μ_g_ix], nrow(urchin_subset)); fill(th_init[μ_p_ix], nrow(urchin_subset))
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
const prep_g_nlyfb = prepare_gradient(nlyfb_urchin, ad_sys, zero(b_init), Constant(th_init), Constant(urchin_subset))
g_nlyfb!(G, b) = gradient!(nlyfb_urchin, G, prep_g_nlyfb, ad_sys, b, Constant(th_init), Constant(urchin_subset))

const sp_ad_sys = AutoSparse(
    ad_sys;
    sparsity_detector=Symbolics.SymbolicsSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
const prep_sp_h_nlyfb = prepare_hessian(nlyfb_urchin, sp_ad_sys, rand(length(b_init)), Constant(th_init), Constant(urchin_subset))

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

# same but in R
R"""
v0e <- expression(-log(2*pi*sigma^2)/2 -
     (sqrt(y) - sqrt(
          exp(w)*exp(exp(g)*a)
     ))^2/(2*sigma^2)
     - log(2*pi) - log(sig.g*sig.p) -
     (g-mu.g)^2/(2*sig.g^2) - (p-mu.p)^2/(2*sig.p^2))

v0 <- deriv(v0e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))

v1e <- expression(-log(2*pi*sigma^2)/2 -
     (sqrt(y) - sqrt(
          exp(p)/exp(g) + exp(p)*(a - log(exp(p) / (exp(g)*exp(w)))/exp(g))          
     ))^2/(2*sigma^2)
     - log(2*pi) - log(sig.g*sig.p) -
     (g-mu.g)^2/(2*sig.g^2) - (p-mu.p)^2/(2*sig.p^2))

v1 <- deriv(v1e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))

# eval \log f_{\theta}(y,b) and the gradient and hessian
# note this assumes that variance params in th are already exp() transformed
# back to constrained scale
lfyb <- function(b,y,a,th) {
     ## evaluate joint p.d.f. of y and b + grad. and Hessian.
     n <- length(y)
     g <- b[1:n]; p <- b[1:n+n]
     am <- (p-g-th[1])/exp(g)
     ind <- a < am
     f0 <- v0(a[ind],y[ind],g[ind],p[ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     f1 <- v1(a[!ind],y[!ind],g[!ind],p[!ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     lf <- sum(f0) + sum(f1)
     g <- matrix(0,n,2) ## extract gradient to g... 
     g[ind,] <- attr(f0,"gradient") ## dlfyb/db 
     g[!ind,] <- attr(f1,"gradient") ## dlfyb/db
     h <- array(0,c(n,2,2)) ## extract Hessian to H... 
     h[ind,,] <- attr(f0,"hessian")
     h[!ind,,] <- attr(f1,"hessian") 
     H <- matrix(0,2*n,2*n)
     for (i in 1:2) for (j in 1:2) {
          indi <- 1:n + (i-1)*n; indj <- 1:n + (j-1)*n
          diag(H[indi,indj]) <- h[,i,j]
     }
     list(lf=lf,g=as.numeric(g),H=H)
}

pdR <- function(H,k.mult=20,tol=.Machine$double.eps^.8) {
     k <- 1
     tol <- tol * norm(H)
     n <- ncol(H)
     while (
          inherits(try(R <- chol(H + (k-1)*tol*diag(n)), silent=TRUE),"try-error")
     ) {
          k <- k * k.mult
     }
     R 
}

# theta on log (unconstrained) scale
llu<-function(theta,vol,age,tol=.Machine$double.eps^.8){
## Laplace approximate log likelihood for urchin model.
    ii <- c(3,5,6)
    theta[ii] <- exp(theta[ii]) ## variance params 
    n <- length(vol)
    if (exists(".inib",envir=environment(llu))) {
        b <- get(".inib",envir=environment(llu))
    } else b <- c(rep(theta[2],n),rep(theta[4],n)); ## init 
    lf <- lfyb(b,vol,age,theta)
    for (i in 1:200) { ## Newton loop...
        R <- pdR(-lf$H) ## R’R = (perturbed) Hessian
        step <- backsolve(R,forwardsolve(t(R),lf$g)) ## Newton 
        conv <- ok <- FALSE
        while (!ok) { ## step halving
            lf1 <- lfyb(b+step,vol,age,theta);
            if (sum(abs(lf1$g)>abs(lf1$lf)*tol)==0) conv <- TRUE
            kk <- 0
            if (!conv&&kk<30&&
                (!is.finite(lf1$lf) || lf1$lf < lf$lf)) {
            step <- step/2;kk <- kk+1
            } else ok <- TRUE
        }
        lf <- lf1;b <- b + step
        if (kk==30||conv) break ## if converged or failed 
    } ## end of Newton loop 
    assign(".inib",b,envir=environment(llu))
    R <- pdR(-lf$H,10)
    ll <- lf$lf - sum(log(diag(R))) + log(2*pi)*n
    return(list(lf=lf, R=R, H=lf$H, b=b))
}
"""

# what to do for Laplace,
# return from llu all the parts used to compute the laplace approx to the log like
# to make sure we are calculating it correctly.

@rput urchin_subset th_init b_init
R"llu_r <- llu(th_init, urchin_subset$vol, urchin_subset$age)"
@rget llu_r

# julia results
nlfyb_optim = optimize(
     b -> nlyfb_urchin(b, th_init, urchin_subset), 
     g_nlyfb!,
     b_init, 
     LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
)
b_hat = Optim.minimizer(nlfyb_optim)
nlfyb_hat = Optim.minimum(nlfyb_optim)

H = hessian(nlyfb_urchin, prep_sp_h_nlyfb, sp_ad_sys, b_hat, Constant(th_init), Constant(urchin_subset))  
R = pdR(H)
logdet(R)

nlfyb_hat + logdet(R) - log(2π)*(length(b_hat)/2)

# r results
llu_r[:lf][:lf]
llu_r[:lf][:H]
llu_r[:R]
llu_r[:H]
llu_r[:b]

llu_r[:lf][:lf] 
sum(log.(diag(llu_r[:R])))

-(llu_r[:lf][:lf] - sum(log.(diag(llu_r[:R]))) + log(2π)*(length(b_hat)/2))

"""
The marginal log-likelihood of the fixed effects L(Θ) = \\int f_{\\Theta}(y,b) db
"""
function marginal_nll(θ)
     # the `llu` function from Wood's R code
     nb = length(b_cache)

     nlfyb_optim = optimize(
          b -> nlyfb_urchin(b, th_init, urchin_subset), 
          g_nlyfb!,
          b_cache, 
          LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
     )
     b_hat = Optim.minimizer(nlfyb_optim)
     nlfyb_hat = Optim.minimum(nlfyb_optim)

     b_cache .= b_hat # updated cached value for next iteration

     H = hessian(nlyfb_urchin, prep_sp_h_nlyfb, sp_ad_sys, b_hat, Constant(th_init), Constant(urchin_subset))  
     R = pdR(H)
     
     return nlfyb_hat + logdet(R) - log(2π)*(nb/2)
end