using LinearAlgebra
using Optim, ForwardDiff, LineSearches

mutable struct Parameters
    beta::Vector{Float64}
    rho1::Float64
    rho2::Float64
    sigmasq::Float64
    dev::Float64
end

"""
    Construct empty Parameters object. Argument `nx` is the number of predictor variables.
"""
Parameters(nx) = Parameters(zeros(nx), 0, 0, 1, Inf)

"""
    Fit the linear network autocorrelation model
"""
function lnam(y, x, W1, W2, opt_nm=opt_bfgs, tol=1E-10)
    nx = size(x,2)
    parm = Parameters(nx)
    
    olddev = Inf
    i = 0
    while abs(parm.dev - olddev) > tol || i<1
        olddev = parm.dev
        parm = estimate(parm, n, x, y, W1, W2, false, opt_nm)
        i += 1
    end
    parm = estimate(parm, n, x, y, W1, W2, true, opt_nm)

    return parm, i
end

"""
    Default optimization of rho parameters, using BFGS with backtracking line search, and forward
mode autodiff for gradients.
"""
function opt_bfgs(n2ll_rho, rho)
    optimize(n2ll_rho, rho, Optim.BFGS(linesearch=LineSearches.BackTracking()), autodiff=:forward)
end

"""
    Conduct a single iterative refinement of a set of initial parameter estimates.

# Arguments
- `parm`: object of type `Parameters`
- `n`: number of data points
- `x`: predictor variables matrix
- `y`: outcome variable vector
- `W1`: network autoregressive matrix
- `W2`: network moving average matrix
- `final`: if true, do not estimate rho
- `opt_rho`: a function that optimizes the marginal nll of rho given starting values of rho
"""
function estimate(parm, n, x, y, W1, W2, final::Bool=false, opt_rho::T=opt_bfgs) where {T<:Function}
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
        n2ll_rho = make_n2ll_rho(n, nx, W1, W2, y, x, parm_new.beta, parm_new.sigmasq)
        rho = [parm_new.rho1,parm_new.rho2]
        rhohat = opt_rho(n2ll_rho, rho)
        parm_new.rho1 = Optim.minimizer(rhohat)[1]
        parm_new.rho2 = Optim.minimizer(rhohat)[2]
    end

    #Calculate model deviance
    parm_new.dev = n*(1+log(2π)+log(parm_new.sigmasq))- 2*(logabsdet(W1a)[1] + logabsdet(W2a)[1])

    # return the parameter list
    return parm_new
end

"""
    Return a function which computes the negative log likelihood.
"""
function make_nll(n,nx,W1,W2,y,x)
    function nll(pars)
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
    return nll
end

"""
    Return a function which computes the negative 2 log likelihood for rho, with beta and sigma fixed.
"""
function make_n2ll_rho(n,nx,W1,W2,y,x,beta,sigmasq)
    nll = make_nll(n,nx,W1,W2,y,x)
    function n2ll_rho(rho)
        pars = [beta...,rho...,sigmasq]
        return 2*nll(pars)
    end
end

"""
    Estimate predicted means, conditional on other effects
"""
function muhat(y,X,W1a,W2a,beta)
    Xb = X * beta
    return W2a * (W1a * y - Xb)
end

"""
    Estimate innovation variance, conditional on other effects
"""
function sigmasqhat(muhat)
    transpose(muhat) * muhat / prod(size(muhat))
end