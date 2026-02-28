from sympy import Expr, Integral, Symbol, nsimplify, srepr, sympify
import torch

from neuralsutra.engine import Engine
from neuralsutra.router import Router
from neuralsutra.vocab import load_vocab


class Compiler:
    """
    Perform a multi-pass transformation on SymPy AST sequences, using the router model
    to dispatch sub-tasks to optimised Vedic kernels.
    """

    def __init__(self, model_path: str, vocab_path: str) -> None:
        self.vocab = load_vocab(vocab_path)
        self.model = Router(vocab_size=len(self.vocab) + 1)

        # Load the trained model (.pth) file
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, node: Expr) -> int:
        """
        Convert a SymPy node to a symbolic expression token sequence and
        predict the best Vedic sutra for the task.
        """
        # Add padding parentheses to the tokens
        tokens = srepr(node).replace("(", " ( ").replace(")", " ) ").split()
        ids = torch.tensor([[self.vocab.get(t, 0) for t in tokens]])

        with torch.no_grad():
            output = self.model(ids)
            return torch.argmax(output, dim=1).item()

    def transform(self, node: Expr, var: Symbol) -> Expr:
        """
        Apply a surgical transformation to each SymPy Integral node.
        """
        if isinstance(node, Integral):
            integrand = node.function

            # Query the neural Router for the mathematical intent
            intent = self.predict(integrand)

            if intent == 1:
                # If there are multiple factors, reduce them recursively
                if integrand.is_Mul:
                    factors = list(integrand.args)

                    def recursive_multiply(args):
                        if len(args) == 1:
                            return args[0]

                        head = Engine.multiply(args[0] * args[1], var)

                        # Recurse with the new result and the remaining factors
                        return recursive_multiply([head] + args[2:])

                    mul_res = recursive_multiply(factors)
                else:
                    mul_res = integrand

                # Re-wrap in Integral and resolve
                return Integral(mul_res, var).doit()
            elif intent == 2:
                # Divide using division kernel
                div_res = Engine.divide(integrand, var)

                return Integral(div_res, var).doit()
            elif intent == 3:
                int_res = Engine.integrate(integrand, var)

                return int_res
            else:
                # Fallback to SymPy
                return node.expand().doit()

        return node

    def compile(self, expr: Expr, var: Symbol, max_passes: int = 10) -> Expr:
        """
        Recursively apply sutras until the expression converges (no Integral nodes left)
        or the structure stabilises (fixed-point iteration).
        """
        # Convert floats to rationals
        expr = nsimplify(sympify(expr), rational=True)

        # Wrap the expression in an Integral object and expand addition only
        current_task = Integral(expr, var).expand(mul=False, multinomial=False)

        last_state = None
        iterations = 0

        while current_task.has(Integral) and iterations < max_passes:
            if last_state == current_task:
                break

            last_state = current_task

            # The replace method handles the tree walking
            current_task = current_task.replace(
                lambda n: isinstance(n, Integral), lambda n: self.transform(n, var)
            )
            iterations += 1

        return current_task
