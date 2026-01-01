from sympy import expand, Mul, simplify, sin

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
