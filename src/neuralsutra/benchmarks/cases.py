from sympy import cos, exp, sin, Symbol, Mul


def get_benchmark_cases():
    """
    Return a dictionary of benchmark expressions classified by
    mathematical structure and computational complexity.
    """
    x = Symbol("x")
    return {
        "POLY_TRANS_MIXED": {
            "expr": -5 * x**12 * cos(x) + 3 * x**2 * exp(x) - (x**3 + 2 * x) / (x + 5),
            "category": "Mixed Terms",
            "description": "Composite expression requiring simultaneous routing to Urdhva Integration and Paravartya Division.",
        },
        "HIGH_DEGREE_SPARSE": {
            "expr": (10 * x**50 + 5 * x**25 + 1) * sin(x),
            "category": "Integration",
            "description": "High-degree sparse polynomial product. Tests the efficiency of recursive coefficient generation.",
        },
        "DUAL_POLY_PRODUCT": {
            "expr": Mul(
                (
                    2 * x**14
                    + x**13
                    + 2 * x**11
                    + x**10
                    + 2 * x**8
                    + x**7
                    + 2 * x**5
                    + x**4
                    + 2 * x**2
                    + x
                ),
                (
                    x**14
                    + x**13
                    + x**12
                    + x**11
                    + x**10
                    + x**9
                    + x**8
                    + x**7
                    + x**6
                    + x**5
                    + x**4
                    + x**3
                    + x**2
                    + x
                    + 1
                ),
                evaluate=False,
            ),
            "category": "Multiplication",
            "description": "Stress test for the Urdhva Tiryagbhyam kernel using two degree-14 polynomials.",
        },
        "NESTED_COMPOSITE": {
            "expr": Mul((x**10 + 5), (x**5 + 2 * x**2 + 1), evaluate=False) * cos(x),
            "category": "Recursive Expansion",
            "description": "Tests multi-pass AST transformation: requires expansion of polynomial sub-products prior to integration.",
        },
        "RATIONAL_FRACTIONAL": {
            "expr": (x**4 + 2) / (x + 2) + (x**6 + 1) / (x - 1),
            "category": "Division",
            "description": "Evaluates Paravartya performance on multi-term rational functions.",
        },
    }
