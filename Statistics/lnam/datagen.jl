# generate large data
using SparseArrays
using Distributions
using LinearAlgebra

n = 5_000

w1 = sprand(Bool, n, n, 0.01)
w2 = sprand(Bool, n, n, 0.01)

x = rand(Normal(),n,5)
r1 = 0.2
r2 = 0.1
sigma = 0.1
beta = rand(Normal(),5)

#Assemble y from its components:
nu = rand(Normal(0,sigma),n)          #Draw the disturbances
e = (I(n) - r2*w2) \ nu
y = (I(n)-r1*w1) \ (x*beta + e)

# new

beta_0 = zeros(size(x,2))
rho1_0 = 0.0
rho2_0 = 0.0
sigmasq_0 = 0.0
dev_0 = Inf

# old

# include("./lnam.jl")

# tol = 1E-10
# nx = size(x,2)

# opt_nm(n2ll_rho, rho) = begin
#     optimize(n2ll_rho, rho, Optim.NelderMead())    
# end

# parm = Parameters(nx)

# olddev = Inf
# i = 0
# while abs(parm.dev - olddev) > tol || i<1
#     olddev = parm.dev
#     parm = estimate(parm, n, x, y, w1, w2, false, opt_nm)
#     i += 1
# end
# parm = estimate(parm, n, x, y, w1, w2, true, opt_nm)

# nll = make_nll(n,nx,w1s,w2s,y,x)
# infomat = FiniteDiff.finite_difference_hessian(nll, [parm.beta...,parm.rho1,parm.rho2,parm.sigmasq])
# acvm = inv(infomat)
