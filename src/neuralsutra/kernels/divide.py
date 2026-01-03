from sympy import Poly, Rational, simplify


def divide(expr, var):
    """
    Perform polynomial division using the Paravartya Yojayet (Transpose and Apply) method.
    """
    try:
        # Split the SymPy expression into numerator and denominator
        num, den = expr.as_numer_denom()

        # Convert the numerator and denominator into polynomial coefficients
        n_coeffs = [Rational(c) for c in Poly(num, var).all_coeffs()]
        d_coeffs = [Rational(c) for c in Poly(den, var).all_coeffs()]

        leading_coeff = d_coeffs[0]

        # Compute the transformed coefficients for division (transpose)
        div_trans = [-Rational(c) for c in d_coeffs[1:]]

        # Copy numerator coefficients to accumulate division result
        res = list(n_coeffs)
        t_len = len(
            div_trans
        )  # t_len is the length of the transposed denominator coefficients
        n_len = len(n_coeffs)
        split = n_len - t_len

        for i in range(split):
            # Normalize the current column by leading_coeff to get the actual quotient digit
            res[i] = res[i] / leading_coeff

            # Apply the normalized digit to the subsequent coefficients
            if res[i] != 0:
                for j in range(t_len):
                    res[i + 1 + j] += res[i] * div_trans[j]

        # Reconstruct the quotient and remainder polynomials
        q_poly = sum(res[i] * var ** (split - 1 - i) for i in range(split))
        r_poly = sum(res[split + i] * var ** (t_len - 1 - i) for i in range(t_len))

        return q_poly + (r_poly / den)
    except Exception:
        # Revert to SymPy division as a safety fallback
        return simplify(expr)
