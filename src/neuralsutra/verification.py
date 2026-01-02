from sympy import diff, simplify


def verify_integration(original_expr, integrated_result, var):
    """
    Verify the mathematical correctness of an integration result.
    Uses the Fundamental Theorem of Calculus: d/dx [Integral(f(x))] == f(x).
    """
    if integrated_result is None:
        return False

    # Differentiate the result using SymPy
    derivative = diff(integrated_result, var)

    # Check for equivalence: (Derivative - Original) should be zero
    error = simplify(derivative - original_expr)
    return error == 0
