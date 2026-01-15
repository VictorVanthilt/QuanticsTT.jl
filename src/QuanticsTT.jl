# small interface for time ordered integrals using quantics
module QuanticsTT
using TensorOperations
# using MatrixAlgebraKit
include("functions.jl")
export QuanticTT, time_ordered_integral_TT, sin_TT, cos_TT, constant_TT, to_TT

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
    # offset::Float64
    # TODO: add a constant field that keeps track of constant offsets
    # a constant offset increases the rank by 1 so it is best to keep track of it separately
    # Make sure that during integration this IS integrated a -> a * x
    function QuanticTT(data)
        return new{eltype(data[1])}(data)
    end
end

function Base.getindex(q::QuanticTT, i::Int)
    return q.data[i]
end

Base.length(q::QuanticTT) = length(q.data)
Base.eachindex(q::QuanticTT) = eachindex(q.data)
Base.lastindex(q::QuanticTT) = lastindex(q.data)

function Base.show(io::IO, q::QuanticTT)
    println(io, "Rank $(rank(q)) QuanticTT of length $(length(q))")
    println(io, "Grid spacing: ", 1 / 2^(length(q)))
    return nothing
end

function rank(q::QuanticTT)
    return size(q[1], 3)
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

Base.:+(qt::QuanticTT, a::Number) = deepcopy(qt) + constant_TT(a, length(qt))

function Base.:*(a::Number, qt::QuanticTT)
    # scale the quantics TT by a number
    newqt = deepcopy(qt)
    newqt[1] .= a * newqt[1]
    return newqt
end

Base.:-(qt1::QuanticTT, qt2::QuanticTT) = qt1 + (-1) * qt2

"""
    (qt1::QuanticTT) * (qt2::QuanticTT)

    Returns a new quantics TT representing qt1(t) * qt2(t).
"""
function Base.:*(qt1::QuanticTT, qt2::QuanticTT)
    mpo = zeros(2, 2, 2) # down up1 up2
    mpo[1, 1, 1] = 1.0
    mpo[2, 2, 2] = 1.0

    # Merge virtual levels into a singular level
    function merge_virtual_levels(T::Array)
        # TODO: check if this is correct
        χl = size(T, 1) * size(T, 2)
        χr = size(T, 4) * size(T, 5)
        return reshape(T, (χl, size(T, 3), χr))
    end

    tensors = map(eachindex(qt1)) do i
        @tensor cur[-1 -2 -3; -4 -5] := qt1[i][-1 1; -4] * qt2[i][-2 2; -5] * mpo[-3; 1 2]
        return merge_virtual_levels(cur)
    end

    return QuanticTT(tensors)
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

    # This encorporated the dx = 1 / 2^N factor
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

"""
    time_ordered_part(qt1::QuanticTT, qt2::QuanticTT)

    Returns a new quantics TT representing the time ordered part
    qt1(t)*∫_{0}^t qt2(x) dx
"""
function time_ordered_part(qt1::QuanticTT, qt2::QuanticTT)
    Iq2 = integrate(qt2)
    return qt1 * Iq2
end

"""
    time_ordered_integral_TT(vqt::Vector{QuanticTT})

    Returns a quantics TT representing the time ordered integral
"""
function time_ordered_integral_TT(vqt::Vector)
    I = time_ordered_part(vqt[2], vqt[1])
    for qt in vqt[3:end]
        I = time_ordered_part(qt, I)
    end
    finalQT = integrate(I)
    return finalQT(1.0 - eps(1.0))
end

"""
    fxf(qt1::QuanticTT, qt2::QuanticTT)

    Returns the integral ∫₀¹ qt1(x) * qt2(x) dx using the quantics TT representations.

    lol the inner product is an integral of the domain of the product of the two functions
"""
function fxf(qt1::QuanticTT, qt2::QuanticTT)
    @assert length(qt1) == length(qt2) "Quantics TT lengths must match for fxf operation"
    @tensor start[-1 -2; -3 -4] := qt1[1][-1 1; -3] * qt2[1][-2 1; -4]
    start *= 0.5
    for i in 2:length(qt1)
        @tensor start[-1 -2; -3 -4] := start[-1 -2; 1 2] * qt1[i][1 3; -3] * qt2[i][2 3; -4]
        start *= 0.5
    end

    return start
end

function to_TT(s::String, omega::Float64, a::Float64, b::Float64, N::Int)
    if s == "sin"
        return sin_TT((b - a) * omega, N; x0 = (-omega * a))
    elseif s == "cos"
        return cos_TT(a, (b - a) * omega, N; x0 = (-omega * a))
    else
        error("Function $s not recognized")
    end
end

# """
#     compress(qt::QuanticTT; tol::Float64 = eps(Float64))

#     Compress the quantics TT using a specified tolerance using SVD truncation.
# """
# function compress(qt::QuanticTT; tol::Float64 = eps(Float64))
#     tensors = deepcopy(qt.data)
#     for i in eachindex(tensors)[1:(end - 1)]
#         @tensor A[-1 -2 -3; -4] := tensors[i][-1 -2; 1] * tensors[i + 1][1 -3; -4]
#         U, S, V, _ = svd_trunc(reshape(A, (size(A, 1) * size(A, 2), size(A, 3) * size(A, 4))); trunc = trunctol(rtol = tol))
#         U = reshape(U, (size(tensors[i], 1), 2, size(U, 2)))
#         V = reshape(V, (size(V, 1), 2, size(tensors[i + 1], 3)))
#         @tensor U[-1 -2; -3] := U[-1 -2; 1] * S[1; -3]
#         tensors[i] = U
#         tensors[i + 1] = V
#     end
#     return QuanticTT(tensors)
# end

end
