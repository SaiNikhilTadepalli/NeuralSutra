import numpy as np
from sympy import Poly


def multiply(expr, var):
    """
    Perform polynomial multiplication using the Urdhva Tiryagbhyam (Vertically and Crosswise) method.
    """
    try:
        # Extract the multiplicative arguments from the SymPy expression
        args = expr.args
        p1 = Poly(args[0], var)
        p2 = Poly(args[1], var)

        # Convert polynomial coefficients into NumPy arrays
        c1 = np.array(p1.all_coeffs(), dtype=float)
        c2 = np.array(p2.all_coeffs(), dtype=float)

        # Perform polynomial multiplication using convolution
        res_coeffs = np.convolve(c1, c2)

        # Compute the degree of the resulting polynomial
        deg = len(res_coeffs) - 1

        # Reconstruct the symbolic polynomial expression
        return sum(round(float(c)) * var ** (deg - i) for i, c in enumerate(res_coeffs))
    except:
        # Revert to SymPy multiplication as a safety fallback
        return expr.expand()
