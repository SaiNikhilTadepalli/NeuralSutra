from sympy import srepr
import torch

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
        Convert a SymPy node to a symbolic expression token sequency and
        predict the best Vedic sutra for the task.
        """
        # Add padding parentheses to the tokens
        tokens = srepr(node).replace("(", " ( ").replace(")", " ) ").split()
        ids = torch.tensor([[self.vocab.get(t, 0) for t in tokens]])

        with torch.no_grad():
            output = self.model(ids)
            return torch.argmax(output, dim=1).item()

    def compile(self, expr, var):
        pass
