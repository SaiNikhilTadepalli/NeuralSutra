from sympy import simplify, sin

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


def test_divide_high_degree(x):
    """Test for correctness for high-degree polynomial division."""
    num = x**4 - 1
    den = x - 1
    expr = num / den

    result = divide(expr, x)
    expected = x**3 + x**2 + x + 1

    assert simplify(result - expected) == 0


def test_divide_negative_coefficients(x):
    """Test for correctness for division with negative signs in the numerator and denominator."""
    num = 2 * x**2 - 7 * x + 3
    den = x - 3
    expr = num / den

    result = divide(expr, x)
    expected = 2 * x - 1

    assert simplify(result - expected) == 0


def test_divide_high_degree_divisor(x):
    """Test for correctness where the denominator is of higher order than (x + a)."""
    num = x**3 + 3 * x**2 + 3 * x + 1
    den = x**2 + 2 * x + 1
    expr = num / den

    result = divide(expr, x)
    expected = x + 1

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
