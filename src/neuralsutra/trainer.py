from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from neuralsutra.router import Router


def train_router(dataset, vocab, model_path, test_size, epochs, lr, weight_decay):
    """Train and validate the Router model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Router(len(vocab) + 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_data, val_data = train_test_split(dataset, test_size=test_size)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for s_expr, label in train_data:
            tokens = s_expr.replace("(", " ( ").replace(")", " ) ").split()
            ids = torch.tensor([[vocab.get(t, 0) for t in tokens]]).to(device)

            optimizer.zero_grad()
            output = model(ids)
            loss = criterion(output, torch.tensor([label]).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate on the held-out dataset
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for s_expr, label in val_data:
                tokens = s_expr.replace("(", " ( ").replace(")", " ) ").split()
                ids = torch.tensor([[vocab.get(t, 0) for t in tokens]]).to(device)
                output = model(ids)

                val_loss += criterion(output, torch.tensor([label]).to(device)).item()
                pred = output.argmax(dim=1).item()
                if pred == label:
                    correct += 1

        avg_train = train_loss / len(train_data)
        avg_val = val_loss / len(val_data)
        accuracy = (correct / len(val_data)) * 100

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {accuracy:.2f}%"
        )

    # Save the model
    save_model(model, model_path)


def save_model(model, path):
    """Save the trained model as a serialized PyTorch state dictionary (.pth) file."""
    torch.save(model.state_dict(), path)
