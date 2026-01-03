from sympy import expand, Mul, simplify, sin, Rational, Float

from neuralsutra.kernels.multiply import multiply


def test_multiply_basic(x):
    """Test for basic correctness."""
    p1 = 2 * x + 3
    p2 = x + 5
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    assert simplify(result - expected) == 0


def test_multiply_high_degree(x):
    """Test for correctness for high-degree polynomials."""
    p1 = x**10 + 2 * x**5 + 1
    p2 = x**8 + x + 3
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    assert simplify(result - expected) == 0


def test_multiply_sparse(x):
    """Test for correctness for sparse polynomials."""
    p1 = x**20 + 1
    p2 = x**15 + x**3
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    assert simplify(result - expected) == 0


def test_multiply_coefficients(x):
    """Test for correctness for polynomials with mixed coefficients and signs."""
    p1 = 3 * x**4 - 2 * x**2 + 7
    p2 = -5 * x**3 + x - 1
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    assert simplify(result - expected) == 0


def test_multiply_fractional(x):
    """Test for correctness with fractional (Rational) coefficients."""
    # This specifically tests if the kernel avoids rounding/truncation
    p1 = Rational(1, 2) * x**2 + Rational(3, 4) * x + 5
    p2 = Rational(2, 3) * x - Rational(1, 5)
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    # The result should be exact, not floating point approximations
    assert simplify(result - expected) == 0
    assert not result.has(Float)


def test_multiply_nested_rational(x):
    """Test for correctness with negative fractions and deeper products."""
    p1 = Rational(-5, 7) * x**3 + Rational(1, 3)
    p2 = Rational(14, 5) * x + 2
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    # Cross-multiplication of (-5/7 * 14/5) should result in exactly -2
    assert simplify(result - expected) == 0


def test_multiply_large_denominators(x):
    """Test that arithmetic precision holds with large denominators."""
    p1 = Rational(1, 123) * x + Rational(1, 456)
    p2 = Rational(1, 789) * x + 1
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    assert simplify(result - expected) == 0


def test_multiply_fallback(x):
    """Test SymPy expansion safety fallback."""
    # (x + 1) * sin(x) is not a polynomial
    p1 = x + 1
    p2 = sin(x)
    expr = Mul(p1, p2, evaluate=False)

    result = multiply(expr, x)
    expected = expand(p1 * p2)

    assert result == expected
    assert result.has(sin)
