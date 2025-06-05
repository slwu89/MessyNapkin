using JuMP, HiGHS
using DataFrames

# T: set of tasks
T = DataFrame(
    i = [:s,1,2,3,4,5,6,7,:f], 
    p = [0,5,4,5,4,4,3,5,0],
    r = [nothing,1,1,3,1,2,4,3,nothing]
)

# E: set of precedence constraints
E = DataFrame(
    i = [:s, 1, 2, 3, 4, :f, :s, 5, 6, 7],
    j = [1, 2, 3, 4, :f, :s, 5, 6, 7, :f],
    H = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
)

# D: set of disjunction constraints
D = crossjoin(T, T, makeunique=true)
rename!(D, :i_1=>:j)
deleteat!(D, findall(D.i .∈ Ref([:s,:f]))) # no start or finish tasks
deleteat!(D, findall(D.j .∈ Ref([:s,:f]))) # no start or finish tasks
deleteat!(D, findall(D.i .== D.j)) # i!=j
deleteat!(D, findall(D.r .!= D.r_1)) # keep (i,j) that use same machine
select!(D, [:i,:j])

# JuMP model and optimizer
model = JuMP.Model(HiGHS.Optimizer)

# 1 / cycle time
@variable(model, τ)

# 5a
for i in eachrow(T)
    if i.i ∈ [:s, :f]
        continue
    else
        @constraint(
            model,
            τ ≤ 1/i.p
        )
    end
end

# 5f and define u (decision variable)
T.u = @variable(model, u[1:nrow(T)] ≥ 0)

# 5e and define K_ij (decision variable)
D.K = @variable(model, [1:nrow(D)], Int)

# 5b
for e in eachrow(E)
    ui = T[findfirst(T.i .== e.i), :u]
    p_i = T[findfirst(T.i .== e.i), :p]
    uj = T[findfirst(T.i .== e.j), :u]
    Hij = e.H
    @constraint(
        model,
        uj + Hij ≥ ui + τ*p_i 
    )
end

# 5c
for d in eachrow(D)
    ui = T[findfirst(T.i .== d.i), :u]
    p_i = T[findfirst(T.i .== d.i), :p]
    uj = T[findfirst(T.i .== d.j), :u]
    Kij = d.K
    @constraint(
        model,
        uj + Kij ≥ ui + τ*p_i
    )
end

# 5d
for d in eachrow(D)
    Kij = d.K
    Kji = D[findfirst(D.i .== d.j .&& D.j .== d.i), :K]
    @constraint(
        model,
        Kij + Kji == 1
    )
end

# obj
@objective(
    model,
    Max,
    τ
)

optimize!(model)

α = value(1/τ)
value.(T.u*α)