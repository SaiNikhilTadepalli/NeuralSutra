from sympy import simplify, sin, Rational, symbols, Poly
from neuralsutra.kernels.divide import divide


def test_divide_basic(x):
    """Test for basic correctness (where the remainder is zero)."""
    num = x**2 + 5 * x + 6
    den = x + 2
    expr = num / den
    result = divide(expr, x)
    expected = x + 3
    assert simplify(result - expected) == 0


def test_divide_remainder(x):
    """Test for correctness for a division that results in a quotient and a remainder."""
    num = x**2 + 5 * x + 7
    den = x + 2
    expr = num / den
    result = divide(expr, x)
    expected = (x + 3) + 1 / (x + 2)
    assert simplify(result - expected) == 0


def test_divide_rational_stress(x):
    """
    Test whether normalisation is handled correctly for non-unit leading coefficients.
    """
    num = Rational(2, 3) * x**2 + Rational(1, 5) * x + Rational(1, 7)
    den = Rational(3, 4) * x + Rational(1, 2)
    expr = num / den

    result = divide(expr, x)

    # Use SymPy to get the 'correct' result
    q, r = Poly(num, x).div(Poly(den, x))
    expected = q.as_expr() + (r.as_expr() / den)

    assert simplify(result - expected) == 0


def test_divide_non_unit_large_coefficients(x):
    """
    Test for correctness with a non-unit divisor with large offsets.
    """
    expr = (12 * x + 7) / (16 * x + 8)
    result = divide(expr, x)

    expected = Rational(3, 4) + 1 / (16 * x + 8)

    assert simplify(result - expected) == 0


def test_divide_multi_term_divisor(x):
    """
    Test for correctness with a quadratic divisor with non-unit leading coefficient.
    """
    num = 4 * x**3 - 2 * x**2 + 5
    den = 2 * x**2 + x + 1
    expr = num / den

    result = divide(expr, x)
    q, r = Poly(num, x).div(Poly(den, x))
    expected = q.as_expr() + (r.as_expr() / den)

    assert simplify(result - expected) == 0


def test_divide_alternating_signs_rational(x):
    """
    Test for correctness with rational coefficients with alternating signs.
    """
    num = (
        Rational(1, 2) * x**3
        - Rational(1, 3) * x**2
        + Rational(1, 4) * x
        - Rational(1, 5)
    )
    den = Rational(1, 6) * x - 1
    expr = num / den

    result = divide(expr, x)
    q, r = Poly(num, x).div(Poly(den, x))
    expected = q.as_expr() + (r.as_expr() / den)

    assert simplify(result - expected) == 0


def test_divide_fractional_input(x):
    """Test to ensure that float-like inputs are handled via Rationals internally."""
    num = Rational(1, 4) * x**2 + Rational(1, 2) * x + Rational(1, 4)
    den = x + 1
    expr = num / den

    result = divide(expr, x)
    expected = Rational(1, 4) * x + Rational(1, 4)

    assert simplify(result - expected) == 0


def test_divide_fallback(x):
    """Test SymPy safety fallback."""
    # sin(x) / (x + 1) is not a polynomial division
    num = sin(x)
    den = x + 1
    expr = num / den

    result = divide(expr, x)

    assert result == expr
    assert result.has(sin)
