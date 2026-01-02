import torch

from neuralsutra.router import Router
from neuralsutra.vocab import load_vocab


class Compiler:
    def __init__(self, model_path, vocab_path):
        self.vocab = load_vocab(vocab_path)
        self.model = Router(vocab_size=len(self.vocab) + 1)

        # Load the trained model (.pth) file
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def compile(self, expr, var):
        pass
