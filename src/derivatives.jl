# SO groups

jacobian(::typeof(inv), g::SO{N}) where {N} = -adjoint(g)

jacobian(::typeof(*), g1::SO{N}, g2::SO{N}) where {N} = inv(adjoint(g2)), I

function left_jacobian(g::SO{3})
    alg = log(g)
    θ² = sum(abs2, alg.θ)
    θ = √θ²
    W = skewsymmetric(alg.θ)
    M1 = (1. - cos(θ))/(θ²) * W
    M2 = (θ - sin(θ))/(θ² * θ) * W^2
    return I(3) + M1 + M2
end

right_jacobian(g::SO{3}) = left_jacobian(g)'


# SE groups

jacobian(::typeof(inv), g::SE{N}) where {N} = -adjoint(g)

ChainRulesCore.frule((_, Δg), ::typeof(inv), g::SE) = inv(g), jacobian(inv, g) * Δg

function ChainRulesCore.rrule(::typeof(inv), g::SE)
    inv_pullback(Δ) = (NoTangent(), jacobian(inv, g)' * Δ)
    inv(g), inv_pullback
end

function jacobian(::typeof(*), g1::SE{N}, g2::SE{N}) where {N}
    R2, t2 = rotation(g2), translation(g2)
    T2 = skewsymmetric(t2)
    z = fill!(similar(R2, N, N), 0)
    J = [R2' -R2'*T2;
           z     R2']
    return J, I(2N)
end

function jacobian(::typeof(⋉), g::SE{N}, x::AbstractVector) where {N}
    R = rotation(g)
    X = skewsymmetric(x)
    return [R -R*X], R
end

function ChainRulesCore.frule((_, Δg, Δx), ::typeof(⋉), g::SE, x::AbstractVector)
    J_g, J_x = jacobian(⋉, g, x)
    return g ⋉ x, J_g * Δg, J_x * Δx
end

function ChainRulesCore.rrule(::typeof(⋉), g::SE, x::AbstractVector)
    J_g, J_x = jacobian(⋉, g, x)
    act_pullback(Δ) = (NoTangent(), J_g' * Δ, J_x' * Δ)
    g ⋉ x, act_pullback
end

jacobian(::typeof(⊕), g::SE, alg::se) =
    jacobian(*, g, exp(alg))[1], right_jacobian(alg)

function ChainRulesCore.frule((_, Δg, Δalg), ::typeof(⊕), g::SE, alg::se)
    J_g, J_alg = jacobian(⊕, g, alg)
    return g ⊕ x, J_g * Δg, J_alg * Δalg
end

function ChainRulesCore.rrule(::typeof(⊕), g::SE, alg::se)
    J_g, J_alg = jacobian(⊕, g, alg)
    oplus_pullback(Δ) = (NoTangent(), J_g' * Δ, J_alg' * Δ)
    g ⊕ x, oplus_pullback
end

# se Lie algebra

function left_jacobian(alg::se{N}) where {N}
    ρ, θ = translation(alg), angle(alg)
    J_l = left_jacobian(so{N}(θ))
    z = fill!(similar(J_l), 0)
    return [J_l Q(ρ, θ);
              z     J_l]
end

right_jacobian(alg::se{N}) where {N} = left_jacobian(se{N}(-translation(alg), -angle(alg)))

function Q(ρ::AbstractVector, θ::AbstractVector)
    θ² = sum(abs2, θ)
    θ_angle = √θ²
    ρ_x, θ_x = skewsymmetric(ρ), skewsymmetric(θ)
    return 0.5*ρ_x +
           (θ_angle - sin(θ_angle)) / θ_angle^3 * (θ_x*ρ_x + ρ_x*θ_x + θ_x*ρ_x*θ_x) +
           -(1 - 0.5*θ_angle^2 - cos(θ_angle)) / θ_angle^4 * (θ_x^2*ρ_x + ρ_x*θ_x^2 - 3θ_x*ρ_x*θ_x) +
           -0.5((1 - 0.5*θ_angle^2 - cos(θ_angle)) / θ_angle^4 - 3(θ_angle - sin(θ_angle) - θ_angle^3/6) / θ_angle^5)*
           (θ_x*ρ_x*θ_x^2 + θ_x^2*ρ_x*θ_x)
end