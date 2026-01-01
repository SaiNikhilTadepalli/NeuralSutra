from sympy import integrate as sympy_integrate, Poly, S


def integrate(expr, var):
    """
    Perform symbolic integration using the Urdhva Tiryagbhyam (Vertically and Crosswise) tabular method.
    """
    try:
        # Extract the constant multiplier from the SymPy expression
        coeff, rest = expr.as_coeff_Mul()

        # Identify the polynomial part (u) and the transcendental part (v)
        args = rest.args if rest.is_Mul else [rest]
        poly_part = next(
            (a for a in args if a.is_polynomial(var)), S.One
        )  # polynomial part
        v = rest / poly_part  # transcendental part

        # If no polynomial found (or u = 1), fallback to SymPy integration
        if poly_part == S.One or not poly_part.has(var):
            return sympy_integrate(expr, var)

        # Convert polynomial part into coefficients
        p = Poly(poly_part, var)
        u_coeffs = list(p.all_coeffs())
        degree = p.degree()

        # Initialize accumulator for the resulting polynomial
        final_poly_coeffs = [S(0)] * (degree + 1)
        curr_u = u_coeffs  # Current row of coefficients in the table
        sign = 1  # Alternating sign
        curr_v_integral = v  # Integral of the transcendental part

        # Integrate recursively while differentiating coefficients
        while curr_u:
            # Integrate the transcendental part for this row
            curr_v_integral = sympy_integrate(curr_v_integral, var)

            # Compute multiplier for this row
            multiplier = (curr_v_integral / v).simplify()

            # Offset to place coefficients correctly in final polynomial
            offset = (degree + 1) - len(curr_u)

            # Add the current row of coefficients to the final accumulator
            for i, c in enumerate(curr_u):
                final_poly_coeffs[offset + i] += sign * c * multiplier

            # Differentiate the coefficients for the next row
            new_u = []
            for i, c in enumerate(curr_u[:-1]):
                power = len(curr_u) - 1 - i
                new_u.append(c * power)
            curr_u = new_u

            # Alternate the sign
            sign *= -1

        # Reconstruct the polynomial from final coefficients and apply constant and transcendental part
        res_poly = sum(c * var ** (degree - i) for i, c in enumerate(final_poly_coeffs))

        return (coeff * res_poly * v).expand()
    except:
        # Revert to SymPy integration as a safety fallback
        return sympy_integrate(expr, var)
