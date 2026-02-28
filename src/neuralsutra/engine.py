from sympy import Expr, Symbol

from neuralsutra.kernels.multiply import multiply
from neuralsutra.kernels.divide import divide
from neuralsutra.kernels.integrate import integrate


class Engine:
    """Central engine for Vedic symbolic operations."""

    @staticmethod
    def multiply(expr: Expr, var: Symbol) -> Expr:
        return multiply(expr, var)

    @staticmethod
    def divide(expr: Expr, var: Symbol) -> Expr:
        return divide(expr, var)

    @staticmethod
    def integrate(expr: Expr, var: Symbol) -> Expr:
        return integrate(expr, var)
