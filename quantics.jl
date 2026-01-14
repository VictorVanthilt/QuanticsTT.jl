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
