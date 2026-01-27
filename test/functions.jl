using Test
using QuanticsTT

# Common parameters
N = 55
ω = 2π
x_vals = collect(0.01:0.001:0.999)
tolerance = 1.0e-12

@testset "Trigonometric function tests" begin

    for (trig_tt, trig) in ((sin_TT, sin), (cos_TT, cos))
        # Test 1: Verify at specific points: trig(ω*x)
        @testset "$(trig) basic evaluation" begin
            tt_func = trig_tt(N; ω = ω)
            for x in x_vals
                val_tt = tt_func(x)
                val_exact = trig(ω * x)
                @test abs(val_tt - val_exact) < tolerance
            end
        end

        # Test 2: x0 offset
        @testset "$(trig) with x0 offset" begin
            x0 = 0.1
            tt_func = trig_tt(N; ω = ω, x0 = x0)

            for x in x_vals
                val_tt = tt_func(x)
                val_exact = trig(ω * (x - x0))
                @test abs(val_tt - val_exact) < tolerance
            end
        end

        # Test 3: Different frequency
        @testset "$(trig) with different frequency" begin
            ω_low = 0.2
            tt_func = trig_tt(N; ω = ω_low)

            for x in x_vals
                val_tt = tt_func(x)
                val_exact = trig(ω_low * x)
                @test abs(val_tt - val_exact) < tolerance
            end
        end

        # Test 6: rescaled shifted offset sin_TT
        @testset "$(trig) with rescaling and offset" begin
            a = 2.1
            b = 3.7
            x0 = 6.7
            tt_func = b * (constant_TT(a, N) + trig_tt(N; ω = ω, x0 = x0))
            for x in x_vals
                val_tt = tt_func(x)
                val_exact = b * (a + trig(ω * (x - x0)))
                @test abs(val_tt - val_exact) < tolerance
            end
        end
    end
end

@testset "sin cos relationship" begin
    ω = 1 / 3 * π
    tt_sin_phase = sin_TT(N; ω = ω, x0 = -π / (2ω))
    tt_cos_direct = cos_TT(N; ω = ω)

    x_test = 0.5
    val_sin = tt_sin_phase(x_test)
    val_cos = tt_cos_direct(x_test)
    @test abs(val_sin - val_cos) < tolerance
end

@testset "Hyperbolic function tests" begin
    for (hyp_tt, hyp) in ((sinh_TT, sinh), (cosh_TT, cosh))
        # Test 1: Verify at specific points: hyp(ω*x)
        @testset "$(hyp) basic evaluation" begin
            tt_func = hyp_tt(N; ω = ω)
            for x in x_vals
                val_tt = tt_func(x)
                val_exact = hyp(ω * x)
                @test abs(val_tt - val_exact) < tolerance
            end
        end

        # Test 2: x0 offset
        @testset "$(hyp) with x0 offset" begin
            x0 = 0.1
            tt_func = hyp_tt(N; ω = ω, x0 = x0)
            for x in x_vals
                val_tt = tt_func(x)
                val_exact = hyp(ω * (x - x0))
                @test abs(val_tt - val_exact) < tolerance
            end
        end

        # Test 3: Different frequency
        @testset "$(hyp) with different frequency" begin
            ω_low = 0.2
            tt_func = hyp_tt(N; ω = ω_low)

            for x in x_vals
                val_tt = tt_func(x)
                val_exact = hyp(ω_low * x)
                @test abs(val_tt - val_exact) < tolerance
            end
        end
    end
end

# Up the tolerance
tolerance = 1.0e-15

@testset "constant_TT tests" begin
    # Test 1: Basic constant representation
    @testset "constant_TT basic evaluation" begin
        a = 5.0
        tt_const = constant_TT(a, N)

        # Constant should return the same value everywhere
        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_const(x)
            @test isapprox(val_tt, a; atol = tolerance)
        end
    end

    # Test 2: Zero constant
    @testset "constant_TT zero value" begin
        tt_zero = constant_TT(0.0, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_zero(x)
            @test isapprox(val_tt, 0.0; atol = tolerance)
        end
    end

    # Test 3: Negative constant
    @testset "constant_TT negative value" begin
        a = -3.5
        tt_neg = constant_TT(a, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_neg(x)
            @test isapprox(val_tt, a; atol = tolerance)
        end
    end

    # Test 4: Different grid sizes
    @testset "constant_TT different grid sizes" begin
        a = 2.0
        for N in [4, 6, 8, 10]
            tt_const = constant_TT(a, N)

            x_test = 0.3
            val_tt = tt_const(x_test)
            @test isapprox(val_tt, a; atol = tolerance)
        end
    end

    # Test 5: Very small constant
    @testset "constant_TT small value" begin
        a = 1.0e-5
        tt_small = constant_TT(a, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_small(x)
            @test isapprox(val_tt, a; atol = tolerance)
        end
    end

    # Test 6: Large constant
    @testset "constant_TT large value" begin
        a = 1.0e5
        tt_large = constant_TT(a, N)

        x_vals = [0.0, 0.5]
        for x in x_vals
            val_tt = tt_large(x)
            @test isapprox(val_tt, a; atol = tolerance)
        end
    end

    # Test 7: Complex constants
    @testset "constant_TT complex value" begin
        a = 2.0 + 3.0im
        tt_complex = constant_TT(a, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_complex(x)
            @test isapprox(val_tt, a; atol = tolerance)
        end
    end
end
