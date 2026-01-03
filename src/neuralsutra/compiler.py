from sympy import Integral, nsimplify, srepr, sympify
import torch

from neuralsutra.engine import Engine
from neuralsutra.router import Router
from neuralsutra.vocab import load_vocab


class Compiler:
    """
    Perform a multi-pass transformation on SymPy AST sequences, using the router model
    to dispatch sub-tasks to optimised Vedic kernels.
    """

    def __init__(self, model_path, vocab_path):
        self.vocab = load_vocab(vocab_path)
        self.model = Router(vocab_size=len(self.vocab) + 1)

        # Load the trained model (.pth) file
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, node):
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

    def transform(self, node, var):
        """
        Apply a surgical transformation to each SymPy Integral node.
        """
        if isinstance(node, Integral):
            integrand = node.function

            # Query the neural Router for the mathematical intent
            intent = self.predict(integrand)

            if intent == 1:
                # Expand using multiplication kernel
                mul_res = Engine.multiply(integrand, var)

                # Re-wrap in Integral and resolve
                return Integral(mul_res, var).doit()
            elif intent == 2:
                # Divide using division kernel
                div_res = Engine.divide(integrand, var)

                return Integral(div_res, var).doit()
            elif intent == 3:
                # Integrate using Urdhva integration kernel
                return Engine.integrate(integrand, var)
            else:
                # Fallback to SymPy
                return node.expand().doit()

        return node

    def compile(self, expr, var, max_passes=3):
        """
        Perform 'surgical integration' by breaking the expression into atomic
        nodes and applying sutras recursively.
        """
        # Convert floats to rationals
        expr = nsimplify(sympify(expr), rational=True)

        # Wrap the expression in an Integral object and expand addition only
        current_task = Integral(expr, var).expand(mul=False, multinomial=False)

        for _ in range(max_passes):
            # Terminate if all Integral nodes have been resolved
            if not current_task.has(Integral):
                break

            def apply_transform(node):
                """Use inner function for better pickling/multiprocessing support."""
                return self.transform(node, var)

            # Walk the tree and apply the transform
            current_task = current_task.replace(
                lambda n: isinstance(n, Integral), apply_transform
            )

        return current_task
