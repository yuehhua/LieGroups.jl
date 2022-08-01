# SO groups and so Lie algebra

Base.exp(alg::so{N,T}) where {N,T<:AbstractMatrix} = SO{N}(exp(alg.θ))
Base.exp(alg::so{N,T}) where {N,T<:AbstractVector} = SO{N}(exp(∧(so{N}, alg.θ)))
Base.log(g::SO{N}) where {N} = so{N}(∨(so{N}, log(rotation(g))))


# SE groups and se Lie algebra

V(θ::AbstractVector) = left_jacobian(so{3}(θ))

function Base.exp(alg::se{N,T}) where {N,T<:AbstractMatrix}
    ρ, θ = translation(alg), angle(alg)
    ρ, θ = ∨(se{N}, ρ, θ)
    return exp(se{N}(ρ, θ))
end

function Base.exp(alg::se{N,T}) where {N,T<:AbstractVector}
    ρ, θ = translation(alg), angle(alg)
    R = rotation(exp(so{N}(θ)))
    t = V(θ) * ρ
    return SE{N}(R, t)
end

function Base.log(g::SE{N}) where {N}
    R, t = rotation(g), translation(g)
    θ = angle(log(SO{N}(R)))
    ρ = inv(V(θ)) * t
    return se{N}(ρ, θ)
end
