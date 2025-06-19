# compare the basic likelihood model
using DataFrames, CSV
using Distributions

using RCall

# load the urchin data
urchin = CSV.read("./Urchin/urchin.csv", DataFrame)
select!(urchin, Not(:id))
urchin.age = convert(Vector{Float64}, urchin.age)

urchin_subset = urchin[1:10:nrow(urchin), :]

# index into a single vector for b (random effects)
const log_g_ix = 1:nrow(urchin_subset)
const log_p_ix = range(start=nrow(urchin_subset)+1, length=nrow(urchin_subset))

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
#     return [
#         let 
#             ll = 0.0
#             v = model_urchin_vol(ω, exp(log_g[i]), exp(log_p[i]), urchin[i, :age])
#             ll += logpdf(Normal(sqrt(v), σ), sqrt(urchin[i, :vol]))
#             ll += logpdf(Normal(θ[μ_g_ix], σ_g), log_g[i])
#             ll += logpdf(Normal(θ[μ_p_ix], σ_p), log_p[i])
#             ll
#         end
#         for i in axes(urchin, 1)
#     ]
     logll = 0.0
     for i in axes(urchin,1)
          v = model_urchin_vol(ω, exp(log_g[i]), exp(log_p[i]), urchin[i, :age])
          logll += logpdf(Normal(sqrt(v), σ), sqrt(urchin[i, :vol]))
          logll += logpdf(Normal(θ[μ_g_ix], σ_g), log_g[i])
          logll += logpdf(Normal(θ[μ_p_ix], σ_p), log_p[i])
     end
     return -logll
end

@rput urchin_subset

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
     list(lf=lf,f0=f0,f1=f1,g=as.numeric(g),H=H)
}
"""

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

th_init_r = copy(th_init)
th_init_r[[3,5,6]] .= exp.(th_init_r[[3,5,6]])

b_init_r = [
    fill(th_init_r[μ_g_ix], nrow(urchin_subset)); fill(th_init_r[μ_p_ix], nrow(urchin_subset))
]

R"ll_r <- lfyb($(b_init_r), urchin_subset$vol, urchin_subset$age, $(th_init_r))"
@rget ll_r

ll_j = nlyfb_urchin(b_init, th_init, urchin_subset)

# what to do for Laplace,
# return from llu all the parts used to compute the laplace approx to the log like
# to make sure we are calculating it correctly.