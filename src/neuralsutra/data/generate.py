import random
from sympy import (
    cos,
    sin,
    tan,
    exp,
    sinh,
    cosh,
    tanh,
    log,
    Mul,
    Add,
    Symbol,
    srepr,
    Rational,
    Pow,
    Expr,
)

x = Symbol("x")


def generate_dataset(samples_per_class: int = 2000) -> list[tuple[str, int]]:
    """
    Generate a synthetic dataset of SymPy AST sequences.
    Classes:
    0: Fallback (Simple polynomials, elementary non-integrable)
    1: Multiply (Polynomial * Polynomial)
    2: Divide (Rational expressions / Polynomial Division)
    3: Integrate (Polynomial * Transcendental - Integration by Parts style)
    """
    dataset = []

    print(f"Generating {samples_per_class * 4} samples...\n")

    def get_coeff(allow_negative: bool = True) -> Rational:
        """Return a random fractional coefficient."""
        num = random.randint(1, 10)
        if allow_negative and random.random() > 0.5:
            num *= -1
        den = random.randint(1, 5)
        return Rational(num, den)

    def get_poly(max_degree: int = 3, min_terms: int = 1) -> Add:
        """Generate a random polynomial."""
        num_terms = random.randint(min_terms, max_degree + 1)
        degrees = random.sample(range(max_degree + 1), num_terms)
        return Add(*[get_coeff() * x**d for d in degrees], evaluate=False)

    def get_transcendental() -> Expr:
        """Return a randomly chosen transcendental function."""
        f = random.choice([sin, cos, exp, tan, sinh, cosh, tanh])
        # Use simple linear arguments for integration logic
        return f(random.choice([1, -1, 2, -2]) * x)

    for _ in range(samples_per_class):
        # Class 0: Fallback
        expr_0 = random.choice(
            [
                get_poly(max_degree=2),
                log(abs(get_coeff() * x + get_coeff())),
                random.choice([sin, cos])(x**2),
                exp(x) / x,
                get_coeff() * x,
            ]
        )
        dataset.append((srepr(expr_0), 0))

        # Class 1: Multiplication
        p1 = get_poly(max_degree=4, min_terms=2)
        p2 = get_poly(max_degree=3, min_terms=2)
        dataset.append((srepr(Mul(p1, p2, evaluate=False)), 1))

        # Class 2: Division
        num_poly = get_poly(max_degree=3, min_terms=1)
        den_poly = get_poly(max_degree=1, min_terms=2)

        # Use evaluate=False to keep the fractional structure in the AST
        div_expr = Mul(num_poly, Pow(den_poly, -1, evaluate=False), evaluate=False)
        dataset.append((srepr(div_expr), 2))

        # Class 3: Integration
        p_a = get_poly(max_degree=3, min_terms=1)
        trans = get_transcendental()

        # Generate double and triple bracket scenarios
        if random.random() > 0.7:
            expr_3 = Mul(get_poly(max_degree=1), p_a, trans, evaluate=False)
        else:
            expr_3 = Mul(p_a, trans, evaluate=False)

        dataset.append((srepr(expr_3), 3))

    random.shuffle(dataset)

    print(f"Successfully generated {len(dataset)} samples.\n")

    return dataset
