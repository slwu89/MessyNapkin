using CSV, LinearAlgebra, Tables
H = CSV.read("./Urchin/H.csv", Tables.matrix, header=0)



function cholesky_check(m)
    mc = try
        cholesky(m).U
    catch
        false
    end
    return mc
end

cholesky_check(H)

A = [4. 12. -16.; 12. 37. -43.; -16. -43. 98.]
cholesky_check(A)


function pdR(H, k_mult=20, tol=2.220446e-16^.8)
    k=1
    tol = tol * opnorm(H, 1)
    n = size(H, 2)
    R = cholesky_check(H + (k-1)*tol*I(n))
    while R isa Bool
        k *= k_mult
        R = cholesky_check(H + (k-1)*tol*I(n))
    end
    return R
end

pdR(H)

