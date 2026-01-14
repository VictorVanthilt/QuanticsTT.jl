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

"""
    integrate(qt::QuanticTT)

    Returns a new quantics TT representing the indefinite integral ∫₀ᵗ f(x) dx.
"""
function integrate(qt::QuanticTT)
    # TODO: figure out normalisation
    # Make heaviside MPO
    # Check page 58-59 of December 25' - January 26' notebook for explanation

    # Merge virtual levels into a singular level
    function merge_virtual_levels(T::Array)
        # TODO: check if this is correct
        χl = size(T, 1) * size(T, 2)
        χr = size(T, 4) * size(T, 5)
        return reshape(T, (χl, size(T, 3), χr))
    end

    # Make the integral MPO
    # Work from right to left
    #                  a  y  x  b
    I = zeros(Float64, 2, 2, 2, 2) # left, down, up, right
    # Flag already activated
    I[2, :, :, 2] .= 1.0 # pass through if already activated
    # Flag not yet activated
    I[2, 2, 1, 1] = 1.0 # activate if y > x
    I[1, 1, 1, 1] = 1.0 # stay inactive if y = x
    I[1, 2, 2, 1] = 1.0 # stay inactive if y = x

    I *= 0.5

    # rightmost tensor: start with flag inactive (vr = (1, 0)ᵀ)
    vr = zeros(Float64, 2, 1)
    vr[1, 1] = 1.0

    @tensor right[-1 -2 -3; -4 -5] := qt[end][-1 1; -4] * I[-2 -3; 1 2] * vr[2; -5]
    right_merged = merge_virtual_levels(right)

    tensors = map(reverse(eachindex(qt)[2:(end - 1)])) do i
        @tensor cur[-1 -2 -3; -4 -5] := qt[i][-1 1; -4] * I[-2 -3; 1 -5]
    end
    tensors = merge_virtual_levels.(tensors)

    # leftmost tensor: capture all (vl = (1, 1))
    vl = zeros(Float64, 1, 2)
    vl[1, 1] = 1.0
    vl[1, 2] = 1.0

    @tensor left[-1 -2 -3; -4 -5] := vl[-2 1] * I[1 -3; 1 -5] * qt[1][-1 1; -4]
    left_merged = merge_virtual_levels(left)
    return QuanticTT([left_merged, reverse(tensors)..., right_merged])
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
