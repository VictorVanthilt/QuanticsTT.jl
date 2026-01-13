# https://physics.stackexchange.com/questions/711726/what-is-the-best-way-of-evaluating-time-ordered-integrals-numerically
module TimeOrderedIntegrals
using HCubature, IterTools

export time_ordered_integral, coefficient_dict

function product_of_functions(fs)
    return vars -> prod(f(v) for (f, v) in zip(fs, vars))
end

function change_of_variables(vars, β)
    n = length(vars)
    transformed_vars = Vector{Number}(undef, n)
    transformed_vars[1] = vars[1]
    for i in 2:n
        transformed_vars[i] = vars[i] * prod(vars[1:i-1]) / β^(i - 1)
    end
    return transformed_vars
end

function jacobian(vars, β)
    n = length(vars)
    return prod([(vars[i] / β)^(n - i) for i in 1:n])
end

function new_f(fs, t₀, t)
    function transformed_f(vars)
        β = t - t₀
        changedvars = [v + t₀ for v in change_of_variables(vars, β)]
        return product_of_functions(fs)(changedvars) * jacobian(vars, β)
    end
end

function time_ordered_integral(fs, t₀, t; verbose = false, kwargs...)
    β = t - t₀
    result = hcubature(new_f(fs, t₀, t), fill(0, length(fs)), fill(β, length(fs)); kwargs...)
    if verbose
        println("Error estimate: ", result[2])
    end
    return result[1]
end

function coefficient_dict(Fs::Vector{Function}, t₀::Number, t₁::Number, N::Int; kwargs...)
    n = length(Fs)
    prefactors = Dict{Tuple{Vararg{Int}}, Number}()
    for i in 1:N
        for combination in IterTools.product(fill(collect(1:n), i)...)
            fs = getindex.(Ref(Fs), combination)
            prefactors[combination] = time_ordered_integral(fs, t₀, t₁; kwargs...)
        end
    end
    return prefactors
end

end
