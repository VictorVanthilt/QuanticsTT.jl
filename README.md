# QuanticsTT.jl

Small Julia module for building quantics tensor-train (TT) representations of simple functions on a $2^N$ grid over $[0, 1)$, and for computing integrals/time-ordered integrals in that representation.

## Quick start

```julia
using QuanticsTT

N = 40
ω = 2π

# Build quantics TT for sin(ω x)
qt = sin_TT(N, ω)

# Evaluate at a point
x = 0.37
val = qt(x)

# Indefinite integral: t -> ∫_0^t sin(ω x) dx
iqt = integrate(qt)
val_int = iqt(x)

# Time-ordered integral for two functions
qt1 = sin_TT(N, ω)
qt2 = cos_TT(N, ω)
val_time_ordered = time_ordered_integral_TT([qt1, qt2])
```

## Functions
- `exp_TT(N; ω, x0)`: $\exp(\omega (x - x0))$
- `sin_TT(N; ω, x0)`: $\sin(\omega (x - x0))$
- `cos_TT(N; ω, x0)`: $\cos(\omega (x - x0))$
- `sinh_TT(N; ω, x0)`: $\sinh(\omega (x - x0))$
- `cosh_TT(N; ω, x0)`: $\cosh(\omega (x - x0))$
- `constant_TT(a, N)`: constant function of amplitude a.

## Functionality
- scalar multiplication, addition, mutliplication
- `integrate(f)`: return a TT for $t \mapsto \int_0^t f(x) dx$.
- `time_ordered_integral_TT([a, b, c])`: $\int_0^tdt_1\int_0^{t_1}dt_2\int_0^{t_2} dt_3 \ a(t_1)b(t_2)c(t_3)$ 