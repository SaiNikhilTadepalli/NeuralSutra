import json
import os


def build_vocab(dataset: list[tuple[str, int]]) -> dict[str, int]:
    """Create a unique ID for every symbolic SymPy expression token found in the dataset."""
    all_tokens = " ".join(
        [d[0].replace("(", " ( ").replace(")", " ) ") for d in dataset]
    ).split()
    unique_tokens = sorted(list(set(all_tokens)))
    vocab = {tok: i + 1 for i, tok in enumerate(unique_tokens)}
    vocab["<UNK>"] = 0

    return vocab


def save_vocab(vocab: dict[str, int], path: str) -> None:
    """Save vocabulary JSON to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)

    print(f"Vocabulary saved to: {path}.")


def load_vocab(path: str) -> dict[str, int]:
    """Load vocabulary JSON from path."""
    with open(path, "r") as f:
        return json.load(f)
