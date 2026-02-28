import torch
from torch import Tensor
import torch.nn as nn


class Router(nn.Module):
    """
    Bi-Directional LSTM for classifying mathematical expressions (tokenized SymPy AST sequences)
    into specific Vedic kernels. Outputs one of four classes: multiply, divide, integrate, fallback.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        super(Router, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # LayerNorm helps with the variance introduced by Float tokens
        self.ln = nn.LayerNorm(hidden_dim * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        embedded = self.embedding(x)

        out, _ = self.lstm(embedded)

        pooled = torch.max(out, dim=1)[0]

        normed = self.ln(pooled)
        return self.fc(normed)
