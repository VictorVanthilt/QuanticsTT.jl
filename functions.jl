"""
    sin_TT(ω::Float64, N::Int; x0::Float64 = 0.0)
    sin_TT(a::Float64, ω::Float64, N::Int, x0::Float64 = 0.0)
    
    Generate an quantics TT representation of (a + ) sin(ω(x - x0)) over [0, 1[ on 2^N evenly spaced gridpoints.
"""
function sin_TT(ω::Float64, N::Int; x0::Float64 = 0.0)
    # Left boundary: [1/(2im), -1/(2im)]
    vl = ones(ComplexF64, 1, 2)
    vl[1, 1] = 1 / (2im)
    vl[1, 2] = -1 / (2im)

    # Right boundary: [1, 1]ᵀ
    vr = ones(ComplexF64, 2, 1)

    # get bitstring for x0
    integerx = floor(Int, x0 * 2^(N))
    xstring = bitstring(integerx)[(end - N + 1):end] # "0110..."
    xstring = [parse(Int, c) for c in xstring] # [0, 1, 1, 0, ...]
    x0_bits = reverse(xstring)

    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, 2, 2, 2) # left, physical, right
        for nα in 1:2 # loop over physical index
            A[1, nα, 1] = exp(1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
            A[2, nα, 2] = exp(- 1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
        end
        return A
    end

    # Absorb boundaries
    @tensor tensors[1][-1 -2 -3] := vl[-1 1] * tensors[1][1 -2 -3]
    @tensor tensors[end][-1 -2 -3] := tensors[end][-1 -2 1] * vr[1 -3]
    return QuanticTT(tensors)
end
function sin_TT(a::Float64, ω::Float64, N::Int; x0::Float64 = 0.0)
    # Left boundary: [1/(2im), -1/(2im), a]
    vl = ones(ComplexF64, 1, 3)
    vl[1, 1] = 1 / (2im)
    vl[1, 2] = -1 / (2im)
    vl[1, 3] = a

    # Right boundary: [1, 1, 1]ᵀ
    vr = ones(ComplexF64, 3, 1)

    # get bitstring for x0
    integerx = floor(Int, x0 * 2^(N))
    xstring = bitstring(integerx)[(end - N + 1):end] # "0110..."
    xstring = [parse(Int, c) for c in xstring] # [0, 1, 1, 0, ...]
    x0_bits = reverse(xstring)

    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, 3, 2, 3) # left, physical, right
        for nα in 1:2 # loop over physical index
            A[1, nα, 1] = exp(1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
            A[2, nα, 2] = exp(- 1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
            A[3, nα, 3] = 1.0
        end
        return A
    end

    # Absorb boundaries
    @tensor tensors[1][-1 -2; -3] := vl[-1; 1] * tensors[1][1 -2; -3]
    @tensor tensors[end][-1 -2; -3] := tensors[end][-1 -2; 1] * vr[1; -3]
    return QuanticTT(tensors)
end

"""
    cos_TT(ω::Float64, N::Int; x0::Float64 = 0.0)
    cos_TT(a::Float64, ω::Float64, N::Int; x0::Float64 = 0.0)

    Generate an quantics TT representation of (a + ) cos(ω(x - x0)) over [0, 1[ on 2^N evenly spaced gridpoints.
"""
function cos_TT(ω::Float64, N::Int; x0::Float64 = 0.0)
    # Left boundary: [1/(2im), 1/(2im)]
    vl = ones(ComplexF64, 1, 2)
    vl[1, 1] = 1 / (2)
    vl[1, 2] = 1 / (2)

    # Right boundary: [1, 1]ᵀ
    vr = ones(ComplexF64, 2, 1)

    # get bitstring for x0
    integerx = floor(Int, x0 * 2^(N))
    xstring = bitstring(integerx)[(end - N + 1):end] # "0110..."
    xstring = [parse(Int, c) for c in xstring] # [0, 1, 1, 0, ...]
    x0_bits = reverse(xstring)


    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, 2, 2, 2) # left, physical, right
        for nα in 1:2 # loop over physical index
            A[1, nα, 1] = exp(1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
            A[2, nα, 2] = exp(-1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
        end
        return A
    end

    # Absorb boundaries
    @tensor tensors[1][-1 -2 -3] := vl[-1 1] * tensors[1][1 -2 -3]
    @tensor tensors[end][-1 -2 -3] := tensors[end][-1 -2 1] * vr[1 -3]
    return QuanticTT(tensors)
end
function cos_TT(a::Float64, ω::Float64, N::Int; x0::Float64 = 0.0)
    # Left boundary: [1/(2im), 1/(2im), a]
    vl = ones(ComplexF64, 1, 3)
    vl[1, 1] = 1 / (2)
    vl[1, 2] = 1 / (2)
    vl[1, 3] = a

    # Right boundary: [1, 1, 1]ᵀ
    vr = ones(ComplexF64, 3, 1)

    # get bitstring for x0
    integerx = floor(Int, x0 * 2^(N))
    xstring = bitstring(integerx)[(end - N + 1):end] # "0110..."
    xstring = [parse(Int, c) for c in xstring] # [0, 1, 1, 0, ...]
    x0_bits = reverse(xstring)

    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, 3, 2, 3) # left, physical, right
        for nα in 1:2 # loop over physical index
            A[1, nα, 1] = exp(1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
            A[2, nα, 2] = exp(-1im * ω * 2.0^(α - 1 - N) * ((nα - 1) - x0_bits[α]))
            A[3, nα, 3] = 1.0
        end
        return A
    end

    # Absorb boundaries
    @tensor tensors[1][-1 -2; -3] := vl[-1; 1] * tensors[1][1 -2; -3]
    @tensor tensors[end][-1 -2; -3] := tensors[end][-1 -2; 1] * vr[1; -3]
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
