using CSV, LinearAlgebra, Tables
H = CSV.read("./Urchin/H.csv", Tables.matrix, header=0)

using RCall

function cholesky_check(m)
    try
        cholesky(m).U
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

R"""
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
"""

R"R_r <- pdR($(H),10)"
@rget R_r

R_j = pdR(H, 10)

