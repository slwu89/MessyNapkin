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

using RCall

# load the urchin data
urchin = CSV.read("./Urchin/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

urchin = urchin[[1,3,5,50,100,142], :]

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

# we'll need to optimize this with Newton's method, so need grad/Hess
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
function lyfbs_urchin(b, s, θ, θ′, urchin)
    return s * lfyb_urchin(b, θ, urchin) + lfyb_urchin(b, θ′, urchin)
end

@rput urchin

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

# eval \log f_{\theta}(y,b) and the gradient and hessian
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

# s \log f_{\theta}(y,b) + \log f_{\theta^{'}}(y,b)
lfybs <- function(s,b,vol,age,th,thp) {
     ## evaluate s log f(y,b;th) + log f(y,b;thp)
     lf <- lfyb(b,vol,age,thp)
     if (s!=0) {
          lfs <- lfyb(b,vol,age,th)
          lf$lf <- lf$lf + s * lfs$lf
          lf$g <- lf$g + s * lfs$g
          lf$H <- lf$H + s * lfs$H
     }
     lf 
}
"""

th = th_init
th_p = th_init + rand(length(th_init))

th_r = deepcopy(th_init)
th_r[[3,5,6]] .= exp.(th_r[[3,5,6]])

th_p_r = deepcopy(th_p)
th_p_r[[3,5,6]] .= exp.(th_p_r[[3,5,6]])

# check that so far we are ok
lfyb_urchin(b_init, th, urchin)
lyfbs_urchin(b_init, 0.3, th, th_p, urchin)

R"lfyb($(b_init), urchin$vol, urchin$age, $(th_r))$lf"
R"lfybs(0.3, $(b_init), urchin$vol, urchin$age, $(th_r), $(th_p_r))$lf"

R"""
laplace <- function(s=0,th,thp,vol,age,b=NULL,tol=.Machine$double.eps^.7) {
     ii <- c(3,5,6) # variance parameters
     thp[ii] <- exp(thp[ii])
     th[ii] <- exp(th[ii])
     n <- length(vol)
     # initialize b
     if (is.null(b)) {
          b <- c(rep(thp[2],n),rep(thp[4],n))
     } 
     lf <- lfybs(s,b,vol,age,th,thp)
     # newton loop to find \hat{b}
     for (i in 1:200) {
          # R'R = fixed Hessian, R upper tri
          R <- pdR(-lf$H)
          step <- backsolve(R,forwardsolve(t(R),lf$g)) ## Newton 
          conv <- ok <- FALSE
          while (!ok) {
               lf1 <- lfybs(s,b+step,vol,age,th,thp)
               if (sum(abs(lf1$g)>abs(lf1$lf)*tol)==0 || sum(b+step!=b)==0) {
                    conv <- TRUE
               }
               kk <- 0
               if (!conv&&kk<30&&(!is.finite(lf1$lf) || lf1$lf < lf$lf)) {
                    step <- step/2;kk <- kk+1
               } else {
                    ok <- TRUE
               }
          }
          dlf <- abs(lf$lf-lf1$lf);lf <- lf1;b <- b + step;
          if (dlf<tol*abs(lf$lf)||conv||kk==30) {
               break
          }
     }
     if (s==0) {
          return(list(g=lfyb(b,vol,age,th)$lf,b=b))
     }
     R <- pdR(-lf$H,10)
     list(b=b,rldetH = sum(log(diag(R))))
}
"""

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

# testing what's going on in the Q function

# first calls laplace with s=0
# this basically needs to find b_hat for lfyb(thp)
# and then return lfyb(b_hat,th)

g_nlfyb!(G, b) = gradient!(nlfyb_urchin, G, prep_g_nlfyb, ad_sys, b, Constant(th_p), Constant(urchin))
H_nlfyb!(H, b) = hessian!(nlfyb_urchin, H, prep_h_nlfyb, sp_ad_sys, b, Constant(th_p), Constant(urchin))
# la_optim = optimize(
#     b -> nlfyb_urchin(b, th_p, urchin), 
#     g_nlfyb!,
#     b_init, 
#     LBFGS(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
# )

la_optim = optimize(
    b -> nlfyb_urchin(b, th_p, urchin), 
    g_nlfyb!,
    H_nlfyb!,
    b_init, 
    Newton(;alphaguess=InitialStatic(scaled=true), linesearch=BackTracking())
)

b_hat = Optim.minimizer(la_optim)
# Optim.minimum(la_optim)
lfyb_urchin(b_hat, th, urchin)

lfyb_urchin(b_init, th_p, urchin)

R"lfybs(s=0,$(b_init),urchin$vol,urchin$age,$(th_r),$(th_p_r))$lf"
R"lfyb($(b_init),urchin$vol,urchin$age,$(th_p_r))$lf"

R"la_r <- laplace(s=0,$(th),$(th_p),urchin$vol,urchin$age,b=$(b_init))"
@rget la_r

lfyb_urchin(la_r[:b], th, urchin)