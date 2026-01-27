using QuanticsTT
using QuadGK

@testset "Integration vs QuadGK" begin
    N = 55
    t_vals = 0.01:0.02:0.99

    @testset "constant integrate" begin
        a = 3.14159
        qt = constant_TT(a, N)
        iq = integrate(qt)
        for t in t_vals
            val_tt = iq(t)
            val_quad = quadgk(x -> a, 0.0, t)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-10)
        end
    end

    @testset "sin integrate" begin
        ω = 2π
        qt = sin_TT(N; ω = ω)
        iq = integrate(qt)
        for t in t_vals
            val_tt = iq(t)
            val_quad = quadgk(x -> sin(ω * x), 0.0, t)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-10)
        end
    end

    @testset "cos integrate" begin
        ω = 2π
        qt = cos_TT(N; ω = ω)
        iq = integrate(qt)
        for t in t_vals
            val_tt = iq(t)
            val_quad = quadgk(x -> cos(ω * x), 0.0, t)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-10)
        end
    end

    @testset "sin with offset and constant integrate" begin
        a = 1.25
        ω = 3.0
        x0 = 0.12
        qt = sin_TT(N; ω = ω, x0 = x0) + constant_TT(a, N)
        iq = integrate(qt)
        for t in t_vals
            val_tt = iq(t)
            val_quad = quadgk(x -> a + sin(ω * (x - x0)), 0.0, t)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-10)
        end
    end

    @testset "time ordered integral is just integral for length 1" begin
        for N in [20, 30, 40, 50, 60]
            qt = sin_TT(N)
            val_tt = time_ordered_integral_TT([qt])
            val_quad = quadgk(t -> sin(t), 0.0, 1.0)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-5)
        end
    end

    @testset "time ordered integral length 2" begin
        # 0 -> 1
        for N in [20, 30, 40, 50, 60]
            qt1 = sin_TT(N)
            qt2 = cos_TT(N)
            val_tt = time_ordered_integral_TT([qt1, qt2])
            val_quad = quadgk(t -> sin(t) * quadgk(s -> cos(s), 0.0, t)[1], 0.0, 1.0)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-5)
        end

        # 0.1 -> 0.9
        for N in [20, 30, 40, 50, 60]
            t0 = 0.1
            t1 = 0.9
            dt = t1 - t0
            qt1 = dt * sin_TT(N, ω = dt, x0 = -t0 / dt)
            qt2 = dt * cos_TT(N, ω = dt, x0 = -t0 / dt)
            val_tt = time_ordered_integral_TT([qt1, qt2])
            val_quad = quadgk(t -> sin(t) * quadgk(s -> cos(s), t0, t)[1], t0, t1)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-5)
        end
    end

    @testset "time ordered integral larger length" begin
        N = 20
        qtc = constant_TT(1.0, N)
        for l in 1:10
            @test isapprox(time_ordered_integral_TT(fill(qtc, l)), 1 / factorial(l), atol = 1.0e-12)
        end
    end
end
