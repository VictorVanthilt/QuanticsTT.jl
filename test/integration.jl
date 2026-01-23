using QuanticsTT
using QuadGK

@testset "integrate function vs QuadGK" begin
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
        qt = sin_TT(ω, N)
        iq = integrate(qt)
        for t in t_vals
            val_tt = iq(t)
            val_quad = quadgk(x -> sin(ω * x), 0.0, t)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-10)
        end
    end

    @testset "cos integrate" begin
        ω = 2π
        qt = cos_TT(ω, N)
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
        qt = sin_TT(a, ω, N; x0 = x0)
        iq = integrate(qt)
        for t in t_vals
            val_tt = iq(t)
            val_quad = quadgk(x -> a + sin(ω * (x - x0)), 0.0, t)[1]
            @test isapprox(val_tt, val_quad; rtol = 1.0e-10)
        end
    end
end
