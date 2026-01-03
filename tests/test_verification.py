import pytest

from sympy import sympify

from neuralsutra.verification import verify_integration


@pytest.mark.parametrize(
    "original, integrated, expected",
    [
        # Basic polynomials
        ("x**2", "x**3/3", True),
        ("x**2", "x**3/2", False),  # Wrong power
        # Transcendental functions
        ("cos(x)", "sin(x)", True),
        ("exp(x)", "exp(x)", True),
        ("1/x", "log(x)", True),
        # Structural vs mathematical equality
        # (x+1)**2 expanded is x**2 + 2*x + 1, verification should handle both
        ("2*(x + 1)", "(x + 1)**2", True),
        (
            "2*x + 2",
            "x**2 + 2*x + 5",
            True,
        ),  # Constant shift shouldn't matter as diff(5) = 0
        # Edge cases
        ("x", None, False),
        ("0", "10", True),  # Derivative of constant is 0
    ],
)
def test_verify_integration_logic(x, original, integrated, expected):
    # Convert strings into SymPy objects
    orig_expr = sympify(original)
    int_res = sympify(integrated) if integrated is not None else None

    assert verify_integration(orig_expr, int_res, x) == expected
