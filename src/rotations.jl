# Algebra Interfaces

abstract type AbstractRotationAlgebra <: AbstractLieAlgebra end

struct so{N,V} <: AbstractLieAlgebra
    θ::V

    function so{N}(x::T) where {N,T<:AbstractVector}
        d = length(x)
        @assert check_dim(so{N}, d)
        return new{N,T}(x)
    end
    
    function so{N}(X::T) where {N,T<:AbstractMatrix}
        @assert isskewsymmetric(X)
        @assert size(X, 1) == N
        return new{N,T}(X)
    end
end

check_dim(::Type{so{N}}, d::Int) where {N} = d == N*(N-1)/2

(==)(alg1::so{N}, alg2::so{N}) where {N} = alg1.θ == alg2.θ
Base.isapprox(alg1::so{N}, alg2::so{N}) where {N} = isapprox(alg1.θ, alg2.θ)

identity(alg::so{N,T}) where {N,T<:AbstractVector} =
    so{N}(fill!(similar(alg.θ), 0))

inv(alg::so{N,T}) where {N,T<:AbstractVector} = so{N}(-alg.θ)

(+)(alg1::so{N}, alg2::so{N}) where {N} = so{N}(alg1.θ + alg2.θ)

∧(alg::so{N,T}) where {N,T<:AbstractVector} = so{N}(∧(so{N}, alg.θ))
∧(alg::so{N,T}) where {N,T<:AbstractMatrix} = alg

∨(alg::so{N,T}) where {N,T<:AbstractVector} = alg
∨(alg::so{N,T}) where {N,T<:AbstractMatrix} = so{N}(∨(so{N}, alg.θ))

Base.Vector(alg::so{N,T}) where {N,T<:AbstractVector} = alg.θ
Base.Vector(alg::so{N,T}) where {N,T<:AbstractMatrix} = ∨(so{N}, alg.θ)

Base.Matrix(alg::so{N,T}) where {N,T<:AbstractVector} = ∧(so{N}, alg.θ)
Base.Matrix(alg::so{N,T}) where {N,T<:AbstractMatrix} = alg.θ


# Group Interfaces

abstract type AbstractRotationGroup <: AbstractLieGroup end

struct SO{N,T} <: AbstractRotationGroup
    A::T

    function SO{N}(X::T) where {N,T<:AbstractMatrix}
        @assert size(X, 1) == N
        return new{N,T}(X)
    end
end

identity(::SO{N}) where {N} = SO{N}(I(N))
identity(::Type{SO{N}}) where {N} = SO{N}(I(N))

inv(g::SO{N}) where {N} = SO{N}(inv(g.A))

function (*)(::SO{M}, ::SO{N}) where {M,N}
    throw(ArgumentError("* operation for SO{$M} and SO{$N} group is not defined."))
end

(*)(g1::SO{N}, g2::SO{N}) where {N} = SO{N}(g1.A * g2.A)

(==)(g1::SO{N}, g2::SO{N}) where {N} = g1.A == g2.A
Base.isapprox(g1::SO{N}, g2::SO{N}) where {N} = isapprox(g1.A, g2.A)

Base.Matrix(g::SO) = g.A

function Base.show(io::IO, g::SO{N}) where N
    print(io, "SO{$N}(A=", g.A, ")")
end

⋉(g::SO{N}, x::T) where {N,T<:AbstractVector} = Matrix(g) * x


# Array Interfaces

function ∧(::Type{so{N}}, alg::AbstractVector) where {N}
    d = length(alg)
    @assert check_dim(so{N}, d)
    if N == 2
        E1 = [0 -1;
              1  0]
        Ω = alg[1]*E1
    elseif N == 3
        E1 = [0 0  0;
              0 0 -1;
              0 1  0]
        E2 = [ 0 0 1;
               0 0 0;
              -1 0 0]
        E3 = [0 -1 0;
              1  0 0;
              0  0 0]
        Ω = alg[1]*E1 + alg[2]*E2 + alg[3]*E3
    else
        throw(ArgumentError("not support."))
    end
    return Ω
end

function ∨(::Type{so{N}}, alg::AbstractMatrix) where {N}
    d = size(alg, 1)
    @assert N == d
    if N == 2
        return [alg[2, 1]]
    elseif N == 3
        return [alg[3, 2], alg[1, 3], alg[2, 1]]
    else
        throw(ArgumentError("not support."))
    end
end


# Connection between groups and algebra

Base.exp(alg::so{N,T}) where {N,T<:AbstractMatrix} = SO{N}(exp(alg.θ))
Base.exp(alg::so{N,T}) where {N,T<:AbstractVector} = SO{N}(exp(∧(so{N}, alg.θ)))
Base.log(g::SO{N}) where {N} = so{N}(∨(so{N}, log(g.A)))
