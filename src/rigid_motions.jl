# Algebra Interfaces

abstract type SpecialEuclideanAlgebra{N} <: AbstractLieAlgebra end

dim(::Type{<:SpecialEuclideanAlgebra{N}}) where {N} = N
dim(::SpecialEuclideanAlgebra{N}) where {N} = N

dof(::Type{<:SpecialEuclideanAlgebra{N}}) where {N} = sum(1:N)
dof(::SpecialEuclideanAlgebra{N}) where {N} = sum(1:N)

struct se{N,T,S} <: SpecialEuclideanAlgebra{N}
    ρ::S
    θ::T

    function se{N}(ρ::T, θ::S) where {N,T<:AbstractVector,S<:AbstractVector}
        check_dof(so{N}, length(θ))
        check_dim(se{N}, length(ρ))
        return new{N,S,T}(ρ, θ)
    end
    
    function se{N}(ρ::T, θ::S) where {N,T<:AbstractVector,S<:AbstractMatrix}
        check_dim(se{N}, size(θ, 1))
        check_skewsymmetric(θ)
        check_dim(se{N}, length(ρ))
        return new{N,S,T}(ρ, θ)
    end
end

se{N}(x::AbstractVector) where {N} = se{N}(x[1:N], x[N+1:end])
se{N}(X::AbstractMatrix) where {N} = se{N}(X[1:N, end], X[1:N, 1:N])

Base.angle(alg::se{N}) where {N} = alg.θ
translation(alg::se{N}) where {N} = alg.ρ

(==)(alg1::se{N}, alg2::se{N}) where {N} = alg1.ρ == alg2.ρ && alg1.θ == alg2.θ
Base.isapprox(alg1::se{N}, alg2::se{N}) where {N} = alg1.ρ ≈ alg2.ρ && alg1.θ ≈ alg2.θ

identity(alg::se{N,T}) where {N,T<:AbstractVector} =
    se{N}(fill!(similar(alg.ρ), 0), fill!(similar(alg.θ), 0))

inv(alg::se{N,T}) where {N,T<:AbstractVector} = se{N}(-translation(alg), -angle(alg))

(+)(alg1::se{N}, alg2::se{N}) where {N} = se{N}(alg1.ρ + alg2.ρ, angle(alg1) + angle(alg2))

∧(alg::se{N,T}) where {N,T<:AbstractVector} = se{N}(∧(se{N}, translation(alg), angle(alg)))
∧(alg::se{N,T}) where {N,T<:AbstractMatrix} = alg

∨(alg::se{N,T}) where {N,T<:AbstractVector} = alg
∨(alg::se{N,T}) where {N,T<:AbstractMatrix} = se{N}(∨(se{N}, translation(alg), angle(alg)))

Base.show(io::IO, alg::se{N}) where {N} =
    print(io, "se{$N}(ρ=", translation(alg), ", θ=", angle(alg), ")")


# Group Interfaces

abstract type SpecialEuclideanGroup{N} <: AbstractLieGroup end

dim(::Type{<:SpecialEuclideanGroup{N}}) where {N} = N
dim(::SpecialEuclideanGroup{N}) where {N} = N

dof(::Type{<:SpecialEuclideanGroup{N}}) where {N} = sum(1:N)
dof(::SpecialEuclideanGroup{N}) where {N} = sum(1:N)

struct SE{N, T} <: SpecialEuclideanGroup{N}
    R
    t

    function SE{N}(R::AbstractMatrix{T}, t::AbstractVector{S}) where {N,T,S}
        @assert size(R, 1) == N
        @assert size(t) == (N, )
        Te = float(promote_type(T, S))
        return new{N, Te}(Te.(R), Te.(t))
    end
end

function SE{N}(A::AbstractMatrix) where {N}
    @assert size(A, 1) == N + 1
    R = A[1:N, 1:N]
    t = A[1:N, end]
    return SE{N}(R, t)
end

rotation(g::SE) = g.R
translation(g::SE) = g.t

identity(::Type{SE{N}}) where {N} = I(N+1) |> SE{N}
identity(::SE{N}) where {N} = I(N+1) |> SE{N}

function inv(g::SE{N}) where {N}
    R, t = rotation(g), translation(g)
    return SE{N}(R', -R'*t)
end

function (*)(::SE{M}, ::SE{N}) where {M,N}
    throw(ArgumentError("+ operation for SE{$M} and SE{$N} group is not defined."))
end

(*)(g1::SE{N}, g2::SE{N}) where {N} = Matrix(g1) * Matrix(g2) |> SE{N}

(==)(g1::SE{N}, g2::SE{N}) where {N} = Matrix(g1) == Matrix(g2)
Base.isapprox(g1::SE{N}, g2::SE{N}) where {N} = isapprox(Matrix(g1), Matrix(g2))

function LinearAlgebra.adjoint(g::SE{N}) where {N}
    R, t = rotation(g), translation(g)
    T = skewsymmetric(t)
    z = fill!(similar(R), 0)
    return [R T*R;
            z   R] |> SE{N}
end

(⊕)(g::SE{N}, alg::se{N}) where {N} = g * exp(alg) |> SE{N}

function ⋉(g::SE{N}, x::AbstractVector) where {N}
    y = Matrix(g) * [x..., 1]
    return y[1:N]
end

function Base.Matrix(g::SE{N}) where {N}
    R, t = rotation(g), translation(g)
    z = fill!(similar(t, 1, N), 0)
    return [R t;
            z 1]
end


# Array Interfaces

function ∧(::Type{se{N}}, ρ::AbstractVector{T}, θ::AbstractVector{T}) where {N,T}
    check_dim(se{N}, length(ρ))
    return ρ, ∧(so{N}, θ)
end

∨(::Type{se{N}}, ρ::AbstractVector, Θ::AbstractMatrix) where {N} = ρ, ∨(so{N}, Θ)

Base.show(io::IO, g::SE{N}) where {N} =
    print(io, "SE{$N}(R=", rotation(g), ", t=", translation(g), ")")
