from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from neuralsutra.router import Router


def train_router(
    dataset, vocab, model_path, test_size=0.2, epochs=3, lr=0.001, weight_decay=1e-5
):
    """Train and validate the Router model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = Router(len(vocab) + 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_data, val_data = train_test_split(dataset, test_size=test_size)
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]

            # Tokenize and convert to tensors
            batch_ids = []
            for s_expr, _ in batch:
                tokens = s_expr.replace("(", " ( ").replace(")", " ) ").split()
                batch_ids.append(torch.tensor([vocab.get(t, 0) for t in tokens]))

            # Pad to make all expressions in this batch the same length
            ids = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True).to(
                device
            )
            labels = torch.tensor([label for _, label in batch]).to(device)

            optimizer.zero_grad()
            output = model(ids)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * ids.size(0)

        # Validate on the held-out dataset
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                v_batch = val_data[i : i + batch_size]

                v_ids_list = []
                for s_expr, _ in v_batch:
                    tokens = s_expr.replace("(", " ( ").replace(")", " ) ").split()
                    v_ids_list.append(torch.tensor([vocab.get(t, 0) for t in tokens]))

                v_ids = torch.nn.utils.rnn.pad_sequence(
                    v_ids_list, batch_first=True
                ).to(device)
                v_labels = torch.tensor([l for _, l in v_batch]).to(device)

                v_output = model(v_ids)
                v_loss = criterion(v_output, v_labels)

                val_loss += v_loss.item() * v_ids.size(0)
                correct += (v_output.argmax(dim=1) == v_labels).sum().item()

        avg_train = train_loss / len(train_data)
        avg_val = val_loss / len(val_data)
        accuracy = (correct / len(val_data)) * 100

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {accuracy:.2f}%"
        )

    # Save the model
    save_model(model, model_path)
    print(f"Model saved to {model_path}")


def save_model(model, path):
    """Save the trained model as a serialized PyTorch state dictionary (.pth) file."""
    torch.save(model.state_dict(), path)
