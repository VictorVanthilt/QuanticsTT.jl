"""
    sin_TT(N::Int; ω::Float64=1.0, x0::Float64 = 0.0)
    
    Generate an quantics TT representation of sin(ω(x - x0)) over [0, 1[ on 2^N evenly spaced gridpoints.
"""
function sin_TT(N::Int; ω::Float64 = 1.0, x0::Float64 = 0.0)
    # Left boundary: [1/(2im) * exp[-iωx0], -1/(2im) * exp[iωx0]]
    vl = ones(ComplexF64, 1, 2)
    vl[1, 1] = (1 / 2im) * exp(-1im * ω * x0)
    vl[1, 2] = (-1 / 2im) * exp(1im * ω * x0)

    # Right boundary: [1, 1]ᵀ
    vr = ones(ComplexF64, 2, 1)

    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, 2, 2, 2) # left, physical, right
        A[1, 1, 1] = 1.0 # nα = 0
        A[2, 1, 2] = 1.0
        A[1, 2, 1] = exp(1im * ω * 2.0^(α - 1 - N))
        A[2, 2, 2] = exp(-1im * ω * 2.0^(α - 1 - N))
        return A
    end

    # Absorb boundaries
    @tensor tensors[1][-1 -2 -3] := vl[-1 1] * tensors[1][1 -2 -3]
    @tensor tensors[end][-1 -2 -3] := tensors[end][-1 -2 1] * vr[1 -3]
    return QuanticTT(tensors)
end

"""
    cos_TT(N::Int; ω::Float64=1.0, x0::Float64 = 0.0)

    Generate an quantics TT representation of (a + ) cos(ω(x - x0)) over [0, 1[ on 2^N evenly spaced gridpoints.
"""
function cos_TT(N::Int; ω::Float64 = 1.0, x0::Float64 = 0.0)
    # Left boundary: [1/2 * exp(-iωx0), 1/2 * exp(iωx0)]
    vl = ones(ComplexF64, 1, 2)
    vl[1, 1] = (1 / 2) * exp(-1im * ω * x0)
    vl[1, 2] = (1 / 2) * exp(1im * ω * x0)

    # Right boundary: [1, 1]ᵀ
    vr = ones(ComplexF64, 2, 1)

    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, 2, 2, 2) # left, physical, right
        A[1, 1, 1] = 1.0 # nα = 0
        A[2, 1, 2] = 1.0
        A[1, 2, 1] = exp(1im * ω * 2.0^(α - 1 - N))
        A[2, 2, 2] = exp(-1im * ω * 2.0^(α - 1 - N))
        return A
    end

    # Absorb boundaries
    @tensor tensors[1][-1 -2 -3] := vl[-1 1] * tensors[1][1 -2 -3]
    @tensor tensors[end][-1 -2 -3] := tensors[end][-1 -2 1] * vr[1 -3]
    return QuanticTT(tensors)
end

"""
    constant_TT(a::Float64, N::Int)

    Generate an quantics TT representation of the constant function f(t) = a over [0, 1[ on 2^N evenly spaced gridpoints.
"""
function constant_TT(a::E, N::Int) where {E <: Number}
    tensors = [ones(E, 1, 2, 1) for _ in 1:N]
    tensors[1] *= a
    return QuanticTT(tensors)
end
