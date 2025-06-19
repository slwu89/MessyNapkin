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

function urchin_biomod(b, θ, urchin)
    # extract fixed effects
    σ = exp(θ[log_σ_ix])
    σ_g = exp(θ[log_σ_g_ix])
    σ_p = exp(θ[log_σ_p_ix])
    ω = exp(θ[log_ω_ix])
    # extract random effects
    log_g = b[log_g_ix]
    log_p = b[log_p_ix]
    # volumes
    v = [model_urchin_vol(ω, exp(log_g[i]), exp(log_p[i]), urchin[i, :age]) for i in axes(urchin, 1)]
    return v
end

# the R impl
@rput urchin_subset

R"""
v0e <- expression(
    exp(w)*exp(exp(g)*a)
)

v0 <- deriv(v0e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))

v1e <- expression(
    exp(p)/exp(g) + exp(p)*(a - log(exp(p) / (exp(g)*exp(w)))/exp(g))          
)

v1 <- deriv(v1e,c("g","p"), hessian=TRUE,function.arg=
     c("a","y","g","p","w","mu.g","sig.g","mu.p",
                                    "sig.p","sigma"))

# eval \log f_{\theta}(y,b) and the gradient and hessian
urchin_biomod <- function(b,y,a,th) {
     ## evaluate joint p.d.f. of y and b + grad. and Hessian.
     n <- length(y)
     g <- b[1:n]; p <- b[1:n+n]
     am <- (p-g-th[1])/exp(g)
     ind <- a < am
     f0 <- v0(a[ind],y[ind],g[ind],p[ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     f1 <- v1(a[!ind],y[!ind],g[!ind],p[!ind],
          th[1],th[2],th[3],th[4],th[5],th[6])
     list(am=am,ind=ind,v=c(f0,f1))
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

R"vols_r <- urchin_biomod($(b_init), urchin_subset$vol, urchin_subset$age, $(th_init))"
@rget vols_r

vols_j = urchin_biomod(b_init, th_init, urchin_subset)

DataFrame(
    R=vols_r[:v],
    J=vols_j
)
