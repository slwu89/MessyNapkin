using JuMP, HiGHS
using DataFrames

T = DataFrame(
    i = [:s,1,2,3,4,5,6,7,:f], 
    p = [0,5,4,5,5,4,3,5,0]
)
E = DataFrame(
    src = [:s, 1, 2, 3, 4, :f, :s, 5, 6, 7],
    tgt = [1, 2, 3, 4, :f, :s, 5, 6, 7, :f],
    H = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
)

model = JuMP.Model(HiGHS.Optimizer)

# cycle time
@variable(model, α)

# dec var: starting time and 1c
T.t = @variable(model, t[1:nrow(T)] ≥ 0)

# 1a
for i in eachrow(T)
    @constraint(
        model,
        α ≥ i.p        
    )
end

# 1b
for e in eachrow(E)
    i = e.src
    j = e.tgt
    ti = T[findfirst(T.i .== i), :t]
    pi = T[findfirst(T.i .== i), :p]
    tj = T[findfirst(T.i .== j), :t]
    Hij = e.H
    @constraint(
        model,
        tj + α*Hij ≥ ti + pi
    )
end

# obj
@objective(
    model,
    Min,
    α
)

optimize!(model)

value(α)
value.(T.t)