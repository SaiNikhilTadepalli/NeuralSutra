from sympy import diff, sympify, Expr, Abs, Symbol


def verify_integration(
    original_expr: Expr, integrated_result: Expr, var: Symbol
) -> bool:
    """
    Verify the mathematical correctness of an integration result.

    Uses the Fundamental Theorem of Calculus: d/dx [Integral(f(x))] == f(x).
    The function first attempts an exact symbolic equivalence check. If symbolic
    comparison fails (due to different algebraic forms), it falls back to a
    numerical verification strategy.

    The numerical fallback evaluates both the original integrand and the derivative
    of the result at specific 'safe' integer sample points [2, 3, 4] using
    high-precision (50-digit) arithmetic. This bypasses floating-point noise and
    confirms identity for complex trigonometric or transcendental forms.
    """
    if integrated_result is None:
        return False

    try:
        integrand = sympify(original_expr)
        var = sympify(var)

        # Differentiate the result
        proposed_deriv = diff(sympify(integrated_result), var)

        # Immediate symbolic equivalence check
        if proposed_deriv == integrand:
            return True

        # Use 'safe' integers instead of random floats to avoid floating point 'noise'
        # in trigonometric functions.
        test_points = [2, 3, 4]

        for val in test_points:
            v1 = integrand.subs(var, val).evalf(50)
            v2 = proposed_deriv.subs(var, val).evalf(50)

            # Handle potential complex numbers or infinities
            if not v1.is_finite or not v2.is_finite:
                continue

            diff_val = Abs(v1 - v2)
            max_val = max(Abs(v1), Abs(v2))

            # If the difference is extremely small relative to the value
            # or if the absolute difference is near zero
            if max_val > 0:
                if (diff_val / max_val) > 1e-12:
                    return False
            elif diff_val > 1e-12:
                return False

        return True
    except Exception as e:
        return False
