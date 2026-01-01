from sympy import sin, cos, exp, simplify, diff

from neuralsutra.kernels.integrate import integrate


def test_integrate_poly_sin(x):
    """Test for correctness for a basic integration by parts."""
    expr = x * sin(x)

    result = integrate(expr, x)

    assert simplify(diff(result, x) - expr) == 0


def test_integrate_poly_exp(x):
    """Test for correctness where the polynomial has a degree greater than 1."""
    expr = x**2 * exp(x)

    result = integrate(expr, x)

    assert simplify(diff(result, x) - expr) == 0


def test_integrate_high_degree(x):
    """Test for correctness for a high-degree polynomial."""
    expr = x**5 * cos(x)

    result = integrate(expr, x)

    assert simplify(diff(result, x) - expr) == 0


def test_integrate_negative_coefficients(x):
    """Test for correctness for negative coefficients."""
    expr = -3 * x**2 * sin(x)

    result = integrate(expr, x)

    assert simplify(diff(result, x) - expr) == 0


def test_integrate_sparse(x):
    """Test for correctness where the polynomial is sparse."""
    expr = (x**3 + 1) * exp(x)

    result = integrate(expr, x)

    assert simplify(diff(result, x) - expr) == 0


def test_integrate_fallback(x):
    """Test SymPy safety fallback."""
    # Not a polynomial * transcendental
    expr = sin(x) * cos(x)

    result = integrate(expr, x)

    assert simplify(diff(result, x) - expr) == 0
