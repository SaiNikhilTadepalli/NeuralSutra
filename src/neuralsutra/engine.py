from neuralsutra.kernels.multiply import multiply
from neuralsutra.kernels.divide import divide
from neuralsutra.kernels.integrate import integrate


class Engine:
    """Central engine for Vedic symbolic operations."""

    @staticmethod
    def multiply(expr, var):
        return multiply(expr, var)

    @staticmethod
    def divide(expr, var):
        return divide(expr, var)

    @staticmethod
    def integrate(expr, var):
        return integrate(expr, var)
