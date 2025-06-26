# ----------------------------------------------------------------------
# test fitting lnam

using CSV, Tables
using LinearAlgebra, SparseArrays 
using Optim, ForwardDiff, LineSearches
using FiniteDiff

include("./lnam.jl")

# ----------------------------------------------------------------------
# read data
w1 = CSV.read("./lnam/data/small/w1.csv", Tables.matrix)
w2 = CSV.read("./lnam/data/small/w2.csv", Tables.matrix)
x = CSV.read("./lnam/data/small/x.csv", Tables.matrix)

beta = CSV.read("./lnam/data/small/beta.csv", Tables.matrix)[:,1]
nu = CSV.read("./lnam/data/small/nu.csv", Tables.matrix)[:,1]
e = CSV.read("./lnam/data/small/e.csv", Tables.matrix)[:,1]
y = CSV.read("./lnam/data/small/y.csv", Tables.matrix)[:,1]

n = length(y)
nx = size(x,2)
rho = [0.2,0.1]
sigmasq = 0.01

tol = 1E-10

# nll = make_nll(n,nx,w1,w2,y,x)
# n2ll_rho = make_n2ll_rho(n,nx,w1,w2,y,x,beta,sigmasq)

# nll([beta...,rho...,sigmasq]) ≈ n2ll_rho(rho)/2

# ----------------------------------------------------------------------
# test with dense matrices
parm = Parameters(nx)

olddev = Inf
i = 0
while abs(parm.dev - olddev) > tol || i<1
    olddev = parm.dev
    parm = estimate(parm, n, x, y, w1, w2, false)
    i += 1
end
parm = estimate(parm, n, x, y, w1, w2, true)

nll = make_nll(n,nx,w1,w2,y,x)
infomat = ForwardDiff.hessian(nll, [parm.beta...,parm.rho1,parm.rho2,parm.sigmasq])
acvm = inv(infomat)
se = sqrt.(diag(acvm))

# ----------------------------------------------------------------------
# test with sparsity
w1s = sparse(w1)
w2s = sparse(w2)

opt_nm(n2ll_rho, rho) = begin
    optimize(n2ll_rho, rho, Optim.NelderMead())    
end

parmsp = Parameters(nx)

olddev = Inf
i = 0
while abs(parmsp.dev - olddev) > tol || i<1
    olddev = parmsp.dev
    parmsp = estimate(parmsp, n, x, y, w1, w2, false, opt_nm)
    i += 1
end
parmsp = estimate(parmsp, n, x, y, w1, w2, true, opt_nm)

nll = make_nll(n,nx,w1s,w2s,y,x)
infomat = FiniteDiff.finite_difference_hessian(nll, [parmsp.beta...,parmsp.rho1,parmsp.rho2,parmsp.sigmasq])
acvm = inv(infomat)
se = sqrt.(diag(acvm))

# ----------------------------------------------------------------------
# MWE: ForwardDiff.gradient does not work well with logabsdet and sparse matrices
# so default of optimizing rho with autodiff gradient will not work
W1s = sparse([1,2,3,5,5],[2,1,2,4,5],ones(Int,5))
W1d = Matrix(W1s)

ForwardDiff.gradient(x -> begin
    return logabsdet(W1s*prod(x))[1]    
end, [0.1,0.2])