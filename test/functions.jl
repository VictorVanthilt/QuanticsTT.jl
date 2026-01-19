using Test
using QuanticsTT

N = 40
x_vals = collect(0.01:0.001:0.999)

@testset "sin_TT tests" begin
    # Test 1: sin_TT with default parameters
    ω = 2π

    # Verify at specific points: sin(ω*x)
    @testset "sin_TT basic evaluation" begin
        tt_sin = sin_TT(ω, N)
        for x in x_vals
            val_tt = tt_sin(x)
            val_exact = sin(ω * x)
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 2: sin_TT with x0 offset
    @testset "sin_TT with x0 offset" begin
        x0 = 0.1
        tt_sin_offset = sin_TT(ω, N; x0 = x0)

        for x in x_vals
            val_tt = tt_sin_offset(x)
            val_exact = sin(ω * (x - x0))
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 3: sin_TT with constant offset
    @testset "sin_TT with constant offset (a + sin)" begin
        a = 3.0
        tt_sin_const = sin_TT(a, ω, N)

        for x in x_vals
            val_tt = tt_sin_const(x)
            val_exact = a + sin(ω * x)
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 4: sin_TT with both offset and constant
    @testset "sin_TT with constant offset and x0" begin
        a = 2.0
        x0 = 0.15
        tt_sin_both = sin_TT(a, ω, N; x0 = x0)

        for x in x_vals
            val_tt = tt_sin_both(x)
            val_exact = a + sin(ω * (x - x0))
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 5: Different frequency
    @testset "sin_TT with different frequency" begin
        ω_low = 0.2
        tt_sin_low = sin_TT(ω_low, N)

        for x in x_vals
            val_tt = tt_sin_low(x)
            val_exact = sin(ω_low * x)
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 6: Rescaling
    @testset "sin_TT with rescaling" begin
        a = 2.1
        b = 3.7
        x0 = 6.7
        tt_sin = b * sin_TT(a, ω, N; x0 = x0)
        for x in x_vals
            val_tt = tt_sin(x)
            val_exact = b * (a + sin(ω * (x - x0)))
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end
end

@testset "cos_TT tests" begin
    # Test 1: cos_TT with default parameters
    ω = 2π
    tt_cos = cos_TT(ω, N)

    # Verify at specific points: cos(ω*x) at x = 0, 0.5, 0.25
    @testset "cos_TT basic evaluation" begin
        for x in x_vals
            val_tt = tt_cos(x)
            val_exact = cos(ω * x)
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 2: cos_TT with x0 offset
    @testset "cos_TT with x0 offset" begin
        x0 = 0.1
        tt_cos_offset = cos_TT(ω, N; x0 = x0)

        for x in x_vals
            val_tt = tt_cos_offset(x)
            val_exact = cos(ω * (x - x0))
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 3: cos_TT with constant offset
    @testset "cos_TT with constant offset (a + cos)" begin
        a = 3.5
        tt_cos_const = cos_TT(a, ω, N)

        for x in x_vals
            val_tt = tt_cos_const(x)
            val_exact = a + cos(ω * x)
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 4: cos_TT with both offset and constant
    @testset "cos_TT with constant offset and x0" begin
        a = 2.5
        x0 = 0.15
        tt_cos_both = cos_TT(a, ω, N; x0 = x0)

        for x in x_vals
            val_tt = tt_cos_both(x)
            val_exact = a + cos(ω * (x - x0))
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 5: Different frequency
    @testset "cos_TT with different frequency" begin
        ω_low = 0.2
        tt_cos_low = cos_TT(ω_low, N)

        for x in x_vals
            val_tt = tt_cos_low(x)
            val_exact = cos(ω_low * x)
            @test abs(val_tt - val_exact) < 1.0e-10
        end
    end

    # Test 6: cos_TT vs sin_TT relationship
    @testset "cos_TT vs sin_TT phase relationship" begin
        ω = 1 / 3 * π
        tt_sin_phase = sin_TT(ω, N; x0 = -π / (2ω))
        tt_cos_direct = cos_TT(ω, N)

        x_test = 0.5
        val_sin = tt_sin_phase(x_test)
        val_cos = tt_cos_direct(x_test)
        @test abs(val_sin - val_cos) < 1.0e-10
    end
end

@testset "constant_TT tests" begin
    # Test 1: Basic constant representation
    @testset "constant_TT basic evaluation" begin
        a = 5.0
        tt_const = constant_TT(a, N)

        # Constant should return the same value everywhere
        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_const(x)
            @test abs(val_tt - a) < 1.0e-10
        end
    end

    # Test 2: Zero constant
    @testset "constant_TT zero value" begin
        tt_zero = constant_TT(0.0, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_zero(x)
            @test abs(val_tt) < 1.0e-10
        end
    end

    # Test 3: Negative constant
    @testset "constant_TT negative value" begin
        a = -3.5
        tt_neg = constant_TT(a, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_neg(x)
            @test abs(val_tt - a) < 1.0e-10
        end
    end

    # Test 4: Different grid sizes
    @testset "constant_TT different grid sizes" begin
        a = 2.0
        for N in [4, 6, 8, 10]
            tt_const = constant_TT(a, N)

            x_test = 0.3
            val_tt = tt_const(x_test)
            @test abs(val_tt - a) < 1.0e-10
        end
    end

    # Test 5: Very small constant
    @testset "constant_TT small value" begin
        N = 8
        a = 1.0e-5
        tt_small = constant_TT(a, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_small(x)
            @test abs(val_tt - a) < 1.0e-10
        end
    end

    # Test 6: Large constant
    @testset "constant_TT large value" begin
        N = 8
        a = 1.0e5
        tt_large = constant_TT(a, N)

        x_vals = [0.0, 0.5]
        for x in x_vals
            val_tt = tt_large(x)
            @test abs(val_tt - a) / abs(a) < 1.0e-10
        end
    end

    # Test 7: Complex constants
    @testset "constant_TT complex value" begin
        N = 8
        a = 2.0 + 3.0im
        tt_complex = constant_TT(a, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_tt = tt_complex(x)
            @test abs(val_tt - a) < 1.0e-10
        end
    end
end

@testset "Cross-function tests" begin
    # Test 1: sin + constant should equal sin_TT with offset
    @testset "sin_TT additive property" begin
        N = 6
        ω = 2π
        a = 2.0

        tt_sin = sin_TT(ω, N)
        tt_const = constant_TT(a, N)
        tt_sin_offset = sin_TT(a, ω, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_sum = tt_sin(x) + tt_const(x)
            val_offset = tt_sin_offset(x)
            @test abs(val_sum - val_offset) < 1.0e-10
        end
    end

    # Test 2: cos + constant should equal cos_TT with offset
    @testset "cos_TT additive property" begin
        N = 6
        ω = 1.0 * π
        a = 1.5

        tt_cos = cos_TT(ω, N)
        tt_const = constant_TT(a, N)
        tt_cos_offset = cos_TT(a, ω, N)

        x_vals = collect(0.01:0.001:0.999)
        for x in x_vals
            val_sum = tt_cos(x) + tt_const(x)
            val_offset = tt_cos_offset(x)
            @test abs(val_sum - val_offset) < 1.0e-10
        end
    end
end
