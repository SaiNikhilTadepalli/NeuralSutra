from sympy import Poly, Rational


def multiply(expr, var):
    """
    Perform exact polynomial multiplication using Urdhva Tiryagbhyam (Vertically and Crosswise) method.
    """
    try:
        # Extract the coefficients of the multiplicative arguments from the SymPy expression
        p1_coeffs = Poly(expr.args[0], var).all_coeffs()
        p2_coeffs = Poly(expr.args[1], var).all_coeffs()

        n, m = len(p1_coeffs), len(p2_coeffs)

        # Initialize the result array
        res_len = n + m - 1
        res_coeffs = [Rational(0)] * res_len

        # Perform polynomial multiplication using convolution
        for i in range(n):
            for j in range(m):
                res_coeffs[i + j] += p1_coeffs[i] * p2_coeffs[j]

        # Compute the degree of the resulting polynomial
        deg = res_len - 1

        # Reconstruct the symbolic polynomial expression
        return sum(c * var ** (deg - i) for i, c in enumerate(res_coeffs))

    except Exception:
        # Revert to SymPy multiplication as a safety fallback
        return expr.expand()
