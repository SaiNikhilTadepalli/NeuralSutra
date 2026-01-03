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
)

x = Symbol("x")


def generate_dataset(samples_per_class=2000):
    """
    Generate a synthetic dataset of SymPy AST sequences.
    Classes: 0: Fallback, 1: Multiply (Urdhva Tiryagbhyam), 2: Divide (Paravartya Yojayet), 3: Integrate (Urdhva Tiryagbhyam tabular method)
    """
    dataset = []
    print(f"Generating {samples_per_class * 4} samples...")

    def get_coeff():
        """Return a random fractional coefficient."""
        num = random.randint(1, 10)
        den = random.randint(1, 5)
        return Rational(num, den)

    def get_transcendental():
        """Return a randomly chosen transcendental function."""
        f = random.choice([sin, cos, exp, tan, sinh, cosh, tanh])
        return f(random.choice([1, -1, 2, -2]) * x)

    for _ in range(samples_per_class):
        # Class 0: Fallback
        expr_0 = random.choice(
            [
                log(abs(get_coeff() * x + get_coeff())),
                random.choice([sin, cos])(x**2),  # Non-linear argument
                exp(x) / x,  # Exponential integral (non-elementary)
            ]
        )
        dataset.append((srepr(expr_0), 0))

        # Class 1: Multiplication
        p1 = Add(
            *[get_coeff() * x**i for i in random.sample(range(5), 3)], evaluate=False
        )
        p2 = Add(
            *[get_coeff() * x**i for i in random.sample(range(4), 2)], evaluate=False
        )
        dataset.append((srepr(Mul(p1, p2, evaluate=False)), 1))

        # Class 2: Division
        factor = x + get_coeff()
        poly = Add(*[get_coeff() * x**i for i in range(random.randint(1, 3))])
        dataset.append((srepr(Mul(poly, factor, evaluate=False) / factor), 2))

        # Class 3: Integration
        p_a = Add(
            *[get_coeff() * x**i for i in random.sample(range(5), 2)], evaluate=False
        )
        p_b = Add(
            *[get_coeff() * x**i for i in random.sample(range(3), 2)], evaluate=False
        )
        trans = get_transcendental()

        if random.random() > 0.5:
            expr_3 = Mul(p_a, p_b, trans, evaluate=False)
        else:
            expr_3 = Mul(
                get_coeff() * x ** random.randint(1, 15), trans, evaluate=False
            )

        dataset.append((srepr(expr_3), 3))

    random.shuffle(dataset)

    print("Dataset generated")

    return dataset
