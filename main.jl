cd(@__DIR__)
include("integrals.jl")
using .TimeOrderedIntegrals
using TensorOperations

# mpo for the function \sum_i a_i exp(b_i t) with t in [0,1[
# To evaluate a function value, multiply t with 2^N, floor to get an integer and express it as a bitstring
# You get the function value by taking the inner product between this time mpo and that bit-string-product-state
function time_mpo(as, bs, N)
    @assert N >= 2 "not supported"
    @assert length(as) == length(bs)

    As = Array{ComplexF64, 3}[]

    for pos in 1:N
        A = zeros(ComplexF64, length(as), 2, length(as))
        for (i, (a, b)) in enumerate(zip(as, bs))
            A[i, 1, i] = 1
            A[i, 2, i] = exp(b * 2.0^(-pos))
        end
        push!(As, A)
    end


    v_left = ones(1, length(as))
    v_right = reshape(as, (length(as), 1))
    @tensor As[1][-1 -2 -3] := v_left[-1, 1] * As[1][1, -2, -3]
    @tensor As[end][-1 -2 -3] := As[end][-1, -2, 3] * v_right[3, -3]

    return As
end

# evaluate a function that is in the form returned by the function above
function evaluate_function_mpo(mpo, x)
    @assert 0 <= x
    @assert x < 1

    N = length(mpo)

    a = ones(1)
    for (loc, d) in enumerate(reverse(digits(Int(floor(2^N * x)), base = 2, pad = N)))
        cur = mpo[loc][:, d + 1, :]
        @tensor a[-1] := a[1] * cur[1, -1]
    end

    return a[1]
end


# combines mpo_1 and mpo_2, representing functions f_1 and f_2 so that the resulting mpo represents the function
# f(t) = f_1(t) \integral_t^1 f_2(t)
function time_ordered_thing(mpo_1, mpo_2)
    @assert length(mpo_1) == length(mpo_2)
    N = length(mpo_1)

    filter = zeros(ComplexF64, 2, 2, 2, 2, 2) # left, down, right, up_1, up_2

    for i in 1:2
        filter[1, i, 1, i, i] = 0.5

        for j in 1:2
            filter[2, i, 2, i, j] = 0.5

            if j > i
                filter[1, i, 2, i, j] = 0.5
            end
        end
    end

    return map(enumerate(zip(mpo_1, mpo_2))) do (loc, (o_1, o_2))
        @tensor o_mpo[-1 -2 -3;-4 -5 -6 -7] := o_1[-1, 1, -5] * o_2[-2, 2, -6] * filter[-3, -4, -7, 1, 2]
        if loc == 1
            o_mpo = o_mpo[:, :, 1:1, :, :, :, :]
        elseif loc == N
            o_mpo = o_mpo[:, :, :, :, :, :, 1:1] + o_mpo[:, :, :, :, :, :, 2:2]
        end

        size_left = size(o_mpo, 1) * size(o_mpo, 2) * size(o_mpo, 3)
        size_right = size(o_mpo, 5) * size(o_mpo, 6) * size(o_mpo, 7)

        reshape(o_mpo, (size_left, 2, size_right))
    end

end

# calculate the time ordered time integral using time_ordered_thing
function time_ordered_mpo_integral(mpos)
    N = length(mpos[1])
    for m in mpos
        @assert length(m) == N
    end
    trivial_mpo = time_mpo([1], [0], N)

    mpos = vcat([trivial_mpo], mpos)
    for i in 1:(length(mpos) - 1)
        mpos[end - i] = time_ordered_thing(mpos[end - i], mpos[end - i + 1])
    end

    return evaluate_function_mpo(mpos[1], 0.0)
end


let
    # sin(x) = (e^(ix) - e^(-ix))/2i
    sin_mpo = time_mpo([1 / (2im), -1 / (2im)], [1im, -1im], 10)
    cos_mpo = time_mpo([1 / (2), 1 / (2)], [1im, -1im], 10)
    trivial_mpo = time_mpo([1], [0], 10)

    # quick sanity check
    for t in 0:0.01:0.99
        @assert abs(evaluate_function_mpo(sin_mpo, t) - sin(t)) < 0.001
        @assert abs(evaluate_function_mpo(cos_mpo, t) - cos(t)) < 0.001
    end

    T = 15.0
    for warmup in 1:10
        println("-----------------------")
        # some actual integration loop
        for N in [5, 10, 20, 40, 80, 200]
            sin_mpo = time_mpo([1 / (2im), -1 / (2im)], [1im * T, -1im * T], N)
            cos_mpo = time_mpo([1 / (2), 1 / (2)], [1im * T, -1im * T], N)
            trivial_mpo = time_mpo([1], [0], N)

            @time @show N, time_ordered_mpo_integral([cos_mpo, sin_mpo, sin_mpo])
        end

        @time @show time_ordered_integral([t -> sin(t * T), t -> sin(t * T), t -> cos(t * T)], 0.0, 1)
    end
end

sin_mpo = time_mpo([1 / (2im), -1 / (2im)], [1im, -1im], 40);
@time evaluate_function_mpo(sin_mpo, 0.946)
