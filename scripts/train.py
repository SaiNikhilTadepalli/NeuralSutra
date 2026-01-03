from neuralsutra.data.generate import generate_dataset
from neuralsutra.trainer import train_router
from neuralsutra.vocab import build_vocab, save_vocab


if __name__ == "__main__":
    # Generate a raw curriculum dataset
    raw_data = generate_dataset()

    # Build vocab
    vocab = build_vocab(raw_data)

    # Train and save the model
    train_router(raw_data, vocab, "models/router.pth")

    # Save the vocab
    save_vocab(vocab, "models/vocab.json")
