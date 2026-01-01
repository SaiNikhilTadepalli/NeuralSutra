import random
from sympy import cos, exp, ln, Mul, S, sin, srepr, Symbol


x = Symbol("x")


def generate_dataset(samples_per_class=500):
    """
    Generate a synthetic dataset of SymPy AST sequences.
    Classes: 0: Fallback, 1: Multiply (Urdhva Tiryagbhyam), 2: Divide (Paravartya Yojayet), 3: Integrate (Urdhva Tiryagbhyam tabular method)
    """
    dataset = []

    for _ in range(samples_per_class):
        # Class 0: Fallback
        dataset.append((srepr(ln(x**2 + 1)), 0))

        # Class 1: Multiplication
        p1 = x ** random.randint(2, 5) + random.randint(1, 5)
        p2 = x ** random.randint(2, 5) + random.randint(1, 5)
        product_expr = Mul(p1, p2, evaluate=False)
        dataset.append((srepr(product_expr), 1))

        # Class 2: Division
        num = x ** random.randint(2, 10) + random.randint(1, 5)
        den = x + random.randint(1, 10)
        dataset.append((srepr(num / den), 2))

        # Class 3: Integration
        deg = random.randint(1, 50)
        p = x**deg
        t = random.choice([sin(x), cos(x), exp(x)])
        dataset.append((srepr(p * t), 3))

    return dataset
