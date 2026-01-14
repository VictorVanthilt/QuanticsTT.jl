# small interface for time ordered integrals using quantics
using TensorOperations

"""
    Struct QuanticTT{E}

    # Fields
    data::Vector{Array{E, 3}}

    A quantics tensor train (TT) representation of a function over [0, 1[ on 2^N evenly spaced gridpoints.
    data[1] corresponds to the least significant bit 2^0.
    data[end] corresponds to the most significant bit 2^(N-1).
"""
struct QuanticTT{E}
    data::Vector{Array{E, 3}}
    function QuanticTT(data)
        return new{eltype(data[1])}(data)
    end
end

function Base.getindex(q::QuanticTT, i::Int)
    return q.data[i]
end

Base.length(q::QuanticTT) = length(q.data)

function Base.show(io::IO, q::QuanticTT)
    println(io, "QuanticTT of length ", length(q))
    println(io, "Grid spacing: ", 1 / 2^(length(q)))
    return nothing
end

"""
    (qt::QuanticTT)(x::Float64)

    Evaluate the quantics TT at x ∈ [0, 1[.
    This muliplies x by 2^N, takes the floor to get an integer, converts that to a bitstring,
    and uses that to contract the quantics TT.
"""
function (qt::QuanticTT)(x::Float64)
    @assert 0 ≤ x < 1 "x out of bounds for quantics representation"
    integerx = floor(Int, x * 2^(length(qt)))
    xstring = bitstring(integerx)[(end - length(qt) + 1):end] # "0110..."
    xstring = [parse(Int, c) for c in xstring] .+ 1 # [1, 2, 2, 1, ...]
    xstring = reverse(xstring) # little endian

    a = ones(1)
    for (pos, char) in enumerate(xstring)
        cur = qt[pos][:, char, :] # impose physical index
        @tensor a[-1] := a[1] * cur[1, -1]
    end
    return only(a)
end

function Base.:+(qt1::QuanticTT, qt2::QuanticTT)
    # perform the direct sum of the two TTs

    @assert length(qt1) == length(qt2) "lengths of quantics TT must match for addition"
    r1 = rank(qt1)
    nrank = r1 + rank(qt2)

    tensors = map(eachindex(qt1)[2:(end - 1)]) do i
        A = zeros(ComplexF64, nrank, 2, nrank)
        A[1:r1, :, 1:r1] = qt1[i][:, :, :]
        A[(r1 + 1):end, :, (r1 + 1):end] = qt2[i][:, :, :]
        return A
    end

    lefttensor = zeros(ComplexF64, 1, 2, nrank)
    lefttensor[1, :, 1:r1] = qt1[1][:, :, :]
    lefttensor[1, :, (r1 + 1):end] = qt2[1][:, :, :]

    righttensor = zeros(ComplexF64, nrank, 2, 1)
    righttensor[1:r1, :, 1] = qt1[end][:, :, :]
    righttensor[(r1 + 1):end, :, 1] = qt2[end][:, :, :]

    return QuanticTT([lefttensor, tensors..., righttensor])
end

include("functions.jl")

function check_accuracy(N, t)
    QT = cos_TT(5.0, 1.0, N)
    actualcos(x) = 5.0 + cos(1.0 * x)
    return abs(QT(t) - actualcos(t))
end

ω = 0.1
@time QT = cos_TT(1.1, ω, 40)

actualcos(x) = 1.1 + cos(ω * x)
@time QT(0.946)

@time check_accuracy(40, 0.946)
