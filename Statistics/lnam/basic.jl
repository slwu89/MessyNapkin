using CSV, Tables
using LinearAlgebra, SparseArrays 
using Optim, ForwardDiff, LineSearches

# notes:
# 1. it seems like In the R code we can compute Xb once for muhat and n2ll.rho

# read data
w1 = CSV.read("./data/w1.csv", Tables.matrix)
w2 = CSV.read("./data/w2.csv", Tables.matrix)
x = CSV.read("./data/x.csv", Tables.matrix)

beta = CSV.read("./data/beta.csv", Tables.matrix)[:,1]
nu = CSV.read("./data/nu.csv", Tables.matrix)[:,1]
e = CSV.read("./data/e.csv", Tables.matrix)[:,1]
y = CSV.read("./data/y.csv", Tables.matrix)[:,1]

n = length(y)
nx = size(x,2)

tol = 1E-10

mutable struct Parameters
    beta::Vector{Float64}
    rho1::Float64
    rho2::Float64
    sigmasq::Float64
    dev::Float64
end

Parameters(nx) = Parameters(zeros(nx), 0, 0, 1, Inf)

parm = Parameters(nx)

# ------------------------------------------------------
# debug iterations
parm = estimate(parm, n, x, y, w1, w2, false)
parm = estimate(parm, n, x, y, w1, w2, false)
parm = estimate(parm, n, x, y, w1, w2, false)


# rho2 starts to diverge on the 3rd iteration.
parm = Parameters(
    [0.9103787,-0.8755734, 1.4188701,-0.4618064,-0.2639024],
    0.1436447,
    0.3566626,
    1.323224,
    219.271
)


# ------------------------------------------------------
# an iteration of estimate  by hand
W1 = deepcopy(w1)
W2 = deepcopy(w2)

parm_new = deepcopy(parm)

# aggregate weight matrices
W1a = I(n) - W1*parm_new.rho1
W2a = I(n) - W2*parm_new.rho2

# assume covariates given, estimate beta | rho
tXtW2aW2a = transpose(x) * transpose(W2a) * W2a
parm_new.beta = (tXtW2aW2a * x) \ (tXtW2aW2a * W1a * y)

#Estimate sigma | beta, rho
parm_new.sigmasq = sigmasqhat(muhat(y,x,W1a,W2a,parm_new.beta))

# (if not final iteration) estimate rho | beta, sigma
n2ll_rho = make_n2ll_rho(n,parm_new.beta,parm_new.sigmasq,W1,W2,y,x)
n2ll_rho([parm_new.rho1,parm_new.rho2])

rho = [parm_new.rho1,parm_new.rho2]
temp = optimize(n2ll_rho, rho, BFGS(linesearch=LineSearches.BackTracking()), autodiff=:forward)
parm_new.rho1 = Optim.minimizer(temp)[1]
parm_new.rho2 = Optim.minimizer(temp)[2]

#Calculate model deviance
parm_new.dev = n*(1+log(2π)+log(parm_new.sigmasq))- 2*(logabsdet(W1a)[1] + logabsdet(W2a)[1])

# ------------------------------------------------------
# the estimation routine
parm = Parameters(nx)

olddev = Inf
i = 0
while abs(parm.dev - olddev) > tol || i<1
    olddev = parm.dev
    parm = estimate(parm, n, x, y, w1, w2, false)
    i += 1
end
parm = estimate(parm, n, x, y, w1, w2, true)


# is locnll the same as n2ll_rho?
n2ll_rho = make_n2ll_rho(n,parm.beta,parm.sigmasq,w1,w2,y,x)
locnll = make_locnll(n,nx,w1,w2,y,x)

n2ll_rho([parm.rho1,parm.rho2])
locnll([parm.beta...,parm.rho1,parm.rho2,parm.sigmasq])

infomat = ForwardDiff.hessian(locnll,[parm.beta...,parm.rho1,parm.rho2,parm.sigmasq])

infomat2 = ForwardDiff.hessian(
    par -> begin
        n2ll_rho = make_n2ll_rho(n,par[1:nx],par[end],w1,w2,y,x)
        return n2ll_rho(par[nx+1:nx+2])
    end,
    [parm.beta...,parm.rho1,parm.rho2,parm.sigmasq]
)

infomat2/2 ≈ infomat

# ------------------------------------------------------
# fns

# Conduct a single iterative refinement of a set of initial parameter estimates
function estimate(parm, n, x, y, W1, W2, final::Bool=false)
    parm_new = deepcopy(parm)

    # aggregate weight matrices
    W1a = LinearAlgebra.I(n) - W1*parm_new.rho1
    W2a = LinearAlgebra.I(n) - W2*parm_new.rho2

    # estimate beta | rho
    tXtW2aW2a = transpose(x) * transpose(W2a) * W2a
    parm_new.beta = (tXtW2aW2a * x) \ (tXtW2aW2a * W1a * y)

    #Estimate sigma | beta, rho
    parm_new.sigmasq = sigmasqhat(muhat(y,x,W1a,W2a,parm_new.beta))

    # (if not final iteration) estimate rho | beta, sigma
    if !final
        n2ll_rho = make_n2ll_rho(n,parm_new.beta,parm_new.sigmasq,W1,W2,y,x)
        rho = [parm_new.rho1,parm_new.rho2]
        temp = optimize(n2ll_rho, rho, BFGS(linesearch=LineSearches.BackTracking()), autodiff=:forward)
        parm_new.rho1 = Optim.minimizer(temp)[1]
        parm_new.rho2 = Optim.minimizer(temp)[2]
    end

    #Calculate model deviance
    parm_new.dev = n*(1+log(2π)+log(parm_new.sigmasq))- 2*(logabsdet(W1a)[1] + logabsdet(W2a)[1])

    # return the parameter list
    return parm_new
end


# fn to make n2ll.rho
function make_n2ll_rho(n,beta,sigmasq,W1,W2,y,x)
    function n2ll_rho(rho)
        # rho1 (AR)
        W1a = I(n) - W1*rho[1]
        W1ay = W1a * y
        adetW1a = abs(det(W1a))

        # rho2 (MA)
        W2a = I(n) - W2*rho[2]
        tpW2a = transpose(W2a) * W2a
        adetW2a = abs(det(W2a))

        # nx
        Xb = x * beta

        # negloglike
        return n*(log(2*π)+log(sigmasq)) + transpose(W1ay-Xb)*tpW2a*(W1ay-Xb)/sigmasq - 2*(log(adetW1a)+log(adetW2a))
    end
    return n2ll_rho
end

# Estimate predicted means, conditional on other effects
function muhat(y,X,W1a,W2a,betahat)
    Xb = X * betahat
    return W2a * (W1a * y - Xb)
end

#Estimate innovation variance, conditional on other effects
function sigmasqhat(muhat)
    transpose(muhat) * muhat / prod(size(muhat))
end

# stuff to compute the Fisher info matrix
function make_locnll(n,nx,W1,W2,y,x)
    function locnll(pars)
        beta = pars[1:nx]
        rho1 = pars[nx+1]
        rho2 = pars[nx+2]
        sigmasq = pars[end]

        W1a = LinearAlgebra.I(n) - W1*rho1
        W1ay = W1a * y
        ladetW1a = logabsdet(W1a)[1]

        W2a = LinearAlgebra.I(n) - W2*rho2
        tpW2a = transpose(W2a) * W2a
        ladetW2a = logabsdet(W2a)[1]

        Xb = x * beta

        return n/2*(log(2π)+log(sigmasq))+ transpose(W1ay-Xb) * tpW2a * (W1ay-Xb)/(2*sigmasq) -ladetW1a-ladetW2a
    end
    return locnll
end