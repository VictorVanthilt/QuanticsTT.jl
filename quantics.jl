# small interface for time ordered integrals using quantics
using TensorKit

struct QuanticTT{E, S}
    data::Vector{TensorKit.TensorMap{E, S, 2, 1}}
    N::Int
    function QuanticTT(data)
        N = length(data)
        return new{eltype(data[1]), spacetype(data[1])}(data, N)
    end
end

function Base.getindex(q::QuanticTT, i::Int)
    return q.data[i]
end

function Base.show(io::IO, q::QuanticTT)
    return println(io, "QuanticTT of length ", q.N)
end

"""
    sin_TT(ω::Float64, N::Int)
    sin_TT(a::Float64, ω::Float64, N::Int)
    
    Generate an quantics TT representation of (a + ) sin(ω t) over [0, 1[ on 2^N evenly spaced gridpoints.
"""
function sin_TT(ω::Float64, N::Int)
    vl = TensorMap([1 / (2im), -1 / (2im)], ℂ^1 ← ℂ^2)
    vr = ones(ComplexF64, ℂ^2 ← ℂ^1)

    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, ℂ^2 ⊗ ℂ^2 ← ℂ^2)
        for nα in 1:2 # loop over physical index
            A[1, nα, 1] = exp(1im * ω * 2.0^(α - 1 - N) * (nα - 1))
            A[2, nα, 2] = exp(- 1im * ω * 2.0^(α - 1 - N) * (nα - 1))
        end
        return A
    end

    @tensor tensors[1][-1 -2; -3] := vl[-1; 1] * tensors[1][1 -2; -3]
    @tensor tensors[end][-1 -2; -3] := tensors[end][-1 -2; 1] * vr[1; -3]
    return QuanticTT(tensors)
end
function sin_TT(a::Float64, ω::Float64, N::Int)
    vl = TensorMap([1 / (2im), -1 / (2im), a], ℂ^1 ← ℂ^3)
    vr = ones(ComplexF64, ℂ^3 ← ℂ^1)

    tensors = map(1:N) do α # loop over sites
        A = zeros(ComplexF64, ℂ^3 ⊗ ℂ^2 ← ℂ^3)
        for nα in 1:2 # loop over physical index
            A[1, nα, 1] = exp(1im * ω * 2.0^(α - 1 - N) * (nα - 1))
            A[2, nα, 2] = exp(- 1im * ω * 2.0^(α - 1 - N) * (nα - 1))
            A[3, nα, 3] = 1.0
        end
        return A
    end

    @tensor tensors[1][-1 -2; -3] := vl[-1; 1] * tensors[1][1 -2; -3]
    @tensor tensors[end][-1 -2; -3] := tensors[end][-1 -2; 1] * vr[1; -3]
    return QuanticTT(tensors)
end

"""
    (qt::QuanticTT)(x::Float64)

    Evaluate the quantics TT at x ∈ [0, 1[.
    This muliplies x by 2^N, takes the floor to get an integer, converts that to a bitstring,
    and uses that to contract the quantics TT.
"""
function (qt::QuanticTT)(x::Float64)
    @assert 0 ≤ x ≤ 1 "x out of bounds for quantics representation"
    integerx = floor(Int, x * 2^qt.N)
    xstring = bitstring(integerx)[(end - qt.N + 1):end] # "0110..."
    xstring = [parse(Int, c) for c in xstring] .+ 1 # [1, 2, 2, 1, ...]
    xstring = reverse(xstring) # little endian

    # Tensors to put on the physical legs
    n0tensor = zeros(scalartype(qt.data[1]), one(ComplexSpace) ← ℂ^2)
    n0tensor[1] = one(scalartype(n0tensor))

    n1tensor = zeros(scalartype(qt.data[1]), one(ComplexSpace) ← ℂ^2)
    n1tensor[2] = one(scalartype(n1tensor))

    ntensor = [n0tensor, n1tensor]
    @tensor answer[-1; -2] := qt[1][-1 1; -2] * ntensor[xstring[1]][1] # evaluate left boundary
    for (pos, char) in enumerate(xstring[2:end])
        @tensor answer[-1; -2] := answer[-1; 1] * qt[pos + 1][1 2; -2] * ntensor[char][2]
    end

    return only(answer.data)
end

function check_accuracy(N, t)
    omega = 1.0
    QT = sin_TT(2.0, omega, N)
    actualsin(x) = 2.0 + sin(omega * x)
    return abs(QT(t) - actualsin(t))
end

ω = 0.1
QT = sin_TT(ω, 40)
actualsin(x) = 2 + sin(ω * x)

@time sin_TT(0.0, 0.1, 40)(0.946)
