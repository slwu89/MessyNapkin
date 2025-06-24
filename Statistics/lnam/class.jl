# class data
using Catlab
using DataMigrations
using StatsBase

# sample the student-class data
students = [
    "Alice", "Bob", "Carol", "Dan", "Erin", "Eve", "Faith",
    "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory",
    "Mike", "Niaj", "Olivia", "Oscar", "Peggy", "Rupert",
    "Sybil", "Trent", "Trudy", "Victor", "Walter", "Wendy"
]

classes = [
    "Economics", "History", "English", "Math", "Biology",
    "Chemistry", "Basket Weaving", "Sports"
]

datamat = zeros(Int, (length(students),length(classes)))

for i in eachindex(students)
    datamat[
        i,
        sample(eachindex(classes), sample(2:4), replace=false)
    ] .= 1
end

# bipartite (2 mode) schema and data
@present SchUndirectedNamedBipartiteGraph <: SchUndirectedBipartiteGraph begin
    Name::AttrType
    student::Attr(V₁,Name)
    class::Attr(V₂,Name)
end

@acset_type UndirectedNamedBipartiteGraph(SchUndirectedNamedBipartiteGraph, index=[:src, :tgt]) <: AbstractUndirectedBipartiteGraph

data_2mode = @acset UndirectedNamedBipartiteGraph{String} begin
    V₁ = length(students)
    student = students
    V₂ = length(classes)
    class = classes
    E = sum(datamat)
    src = getindex.(findall(datamat .== 1),1)
    tgt = getindex.(findall(datamat .== 1),2)
end

# to get student Alice's courses:
data_2mode[incident(data_2mode, "Alice", [:src,:student]),[:tgt,:class]]

to_graphviz(
    data_2mode, 
    node_labels=(:student,:class)
    # graph_attrs=Dict(:dpi=>"72",:size=>"",:ratio=>"fill")
)

# data migration 1 mode
M = @migration SchLabeledGraph SchUndirectedNamedBipartiteGraph begin
    V => V₁
    E => @join begin
        (s1, s2)::V₁
        (e1, e2)::E
        c::V₂
        src(e1) == s1
        tgt(e1) == c
        src(e2) == s2
        tgt(e2) == c
    end
    src => e1 ⋅ src
    tgt => e2 ⋅ src
    Label => Name
    label => student
end

data_1mode = migrate(LabeledGraph{String}, data_2mode, M)

to_graphviz(data_1mode, node_labels=:label)


# ----------------------------------------------------------------------
# smaller test data

# 2 mode (bipartite) to 1 mode (graph)
data_2mode = @acset UndirectedNamedBipartiteGraph{String} begin
    V₁ = 5
    student = ["Frank","Alice","Bob","Dan","Carol"]
    V₂ = 3
    class = ["History","Math","Biology"]
    E = 6
    src = [1,2,3,3,4,5]
    tgt = [1,1,1,2,2,3]
end

to_graphviz(data_2mode, node_labels=(:student,:class))

data_1mode = migrate(LabeledGraph{String}, data_2mode, M)

# remove self loops
rem_parts!(
    data_1mode,
    :E,
    findall(data_1mode[:,:src] .== data_1mode[:,:tgt])
)

to_graphviz(data_1mode, node_labels=:label)



# ----------------------------------------------------------------------
# symmetric graph? not actually that useful

@present SchSymmetricLabeledGraph <: SchSymmetricGraph begin
    Label::AttrType
    label::Attr(V,Label)
end

@acset_type SymmetricLabeledGraph(SchSymmetricLabeledGraph, index=[:src]) <: AbstractSymmetricGraph

# data migration 1 mode
M2 = @migration SchSymmetricLabeledGraph SchLabeledGraph begin
    V => V
    E => @cases (fwd::E; rev::E)
    src => (fwd => src; rev => tgt)
    tgt => (fwd => tgt; rev => src)
    inv => (fwd => rev; rev => fwd)
    Label => Label
    label => label
end

data_1mode_symm = migrate(SymmetricLabeledGraph{String}, data_1mode, M2)

to_graphviz(data_1mode_symm, node_labels=:label)