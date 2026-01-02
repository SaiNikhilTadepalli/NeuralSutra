import torch
import torch.nn as nn


class Router(nn.Module):
    """
    Bi-Directional LSTM for classifying mathematical expressions (tokenized SymPy AST sequences)
    into specific Vedic kernels. Outputs one of four classes: multiply, divide, integrate, fallback.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=4):
        super(Router, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc(self.dropout(hidden_cat))
        return out
