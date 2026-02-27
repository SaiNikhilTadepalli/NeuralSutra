from sympy import cos, exp, sin, sinh, cosh, tan, log, Symbol, Mul, Rational


def get_benchmark_cases():
    """
    Return a dictionary of benchmark SymPy expressions to test the performance of NeuralSutra.
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
        "HYPERBOLIC_CYCLE": {
            "expr": (x**8 + 4 * x**6 - 3 * x**2) * sinh(x) + (x**3) * cosh(x),
            "category": "New Transcendentals",
            "description": "Evaluates the Urdhva kernel on Hyperbolic functions, testing the extended derivative cycle handling.",
        },
        "TRIPLE_CHAIN_PRODUCT": {
            "expr": Mul(
                (x**2 + 3 * x + 1), (x**2 - 3 * x + 1), (x**4 + 1), evaluate=False
            ),
            "category": "Deep Multiplication",
            "description": "Requires sequential application of Urdhva Tiryagbhyam across three distinct polynomial factors.",
        },
        "LARGE_OFFSET_DIVISION": {
            "expr": (x**9 + 12345 * x**5 - 67890) / (x - 15),
            "category": "Stress Division",
            "description": "Tests Paravartya Yojayet on inputs with large constant offsets, challenging arithmetic overflow boundaries.",
        },
        "NON_VEDIC_FALLBACK": {
            "expr": x**3 * log(x) + tan(x**2),
            "category": "Router Accuracy",
            "description": "Control case: Should be correctly identified as 'Fallback' (Class 0) due to log and non-linear arguments.",
        },
        "FRACTIONAL_COEFF_URDHVA": {
            "expr": Mul(
                (Rational(1, 2) * x**2 + Rational(3, 4) * x + 5),
                (Rational(2, 3) * x**2 - Rational(1, 5)),
                evaluate=False,
            ),
            "category": "Fractional Multiplication",
            "description": "Tests Urdhva Tiryagbhyam with non-integer Rational coefficients to ensure cross-multiplication maintains exactness.",
        },
        "RATIONAL_TRIG_INTEGRATION": {
            "expr": (Rational(1, 10) * x**4 + Rational(2, 3) * x**2) * sin(x),
            "category": "Integration",
            "description": "Integration of a polynomial with rational coefficients against a transcendental function.",
        },
        "FRACTIONAL_DIVISION": {
            "expr": (Rational(1, 4) * x**3 + Rational(5, 2) * x - 1)
            / (x - Rational(1, 2)),
            "category": "Division",
            "description": "Paravartya Yojayet division with fractional coefficients and a fractional divisor offset.",
        },
    }
