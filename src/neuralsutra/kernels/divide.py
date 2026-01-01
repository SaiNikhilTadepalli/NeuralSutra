import numpy as np
from sympy import Poly, simplify


def divide(expr, var):
    """
    Perform polynomial division using the Paravartya Yojayet (Transpose and Apply) method.
    """
    try:
        # Split the SymPy expression into numerator and denominator
        num, den = expr.as_numer_denom()

        # Convert the numerator and denominator into polynomial coefficients
        n_coeffs = Poly(num, var).all_coeffs()
        d_coeffs = Poly(den, var).all_coeffs()

        # Convert polynomial coefficients into NumPy arrays
        n_coeffs = np.array(n_coeffs, dtype=float)
        d_coeffs = np.array(d_coeffs, dtype=float)

        # Compute the transformed coefficients for division
        div_trans = -d_coeffs[1:] / d_coeffs[0]

        # Copy numerator coefficients to accumulate division result
        res = n_coeffs.copy()
        t_len = len(
            div_trans
        )  # t_len is the length of the transposed denominator coefficients

        # Iterate over numerator coefficients except the last t_len elements
        for i in range(len(n_coeffs) - t_len):
            if res[i] != 0:
                # Apply the transformation to subsequent coefficients
                res[i + 1 : i + 1 + t_len] += res[i] * div_trans

        # Split result array into quotient and remainder parts
        split = len(n_coeffs) - t_len

        # Reconstruct the quotient and remainder polynomials
        q_poly = sum(c * var ** (split - 1 - i) for i, c in enumerate(res[:split]))
        r_poly = sum(c * var ** (t_len - 1 - i) for i, c in enumerate(res[split:]))

        return q_poly + (r_poly / den)
    except:
        # Revert to SymPy division as a safety fallback
        return simplify(expr)
