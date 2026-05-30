import argparse
import json
from pathlib import Path


def flatten_vocab(vocab_data):
    flat = {}
    for category, entries in vocab_data.items():
        if isinstance(category, str) and category.startswith("_"):
            continue
        if not isinstance(entries, dict):
            raise ValueError(
                f"Expected category {category} to contain a dict of tokens."
            )
        for token, idx in entries.items():
            if token in flat:
                raise ValueError(f'Duplicate token "{token}" found in vocab.json.')
            flat[token] = idx
    return flat


def validate_flat_vocab(flat_vocab):
    ids = list(flat_vocab.values())
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate token ids found in flattened vocab.")
    if min(ids) != 0:
        print("Warning: lowest token id is not 0.")


def build_embedding_matrix(vocab_size, hidden_dim=128, zero_init=False, seed=42):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required to build the embedding matrix. Install numpy and retry."
        ) from exc

    if zero_init:
        return np.zeros((vocab_size, hidden_dim), dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocab_size, hidden_dim), dtype=np.float32) * 0.02


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild tokenizer vocabulary and embedding matrix from tokenizer/vocab.json."
    )
    parser.add_argument(
        "--vocab", default="tokenizer/vocab.json", help="Path to vocab.json."
    )
    parser.add_argument(
        "--out-flat",
        default="tokenizer/vocab_flat.json",
        help="Output flattened vocabulary JSON file.",
    )
    parser.add_argument(
        "--out-ids",
        default="tokenizer/id_to_token.json",
        help="Output id-to-token JSON file.",
    )
    parser.add_argument(
        "--out-emb",
        default="tokenizer/embedding.npy",
        help="Output token embedding matrix file.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the embedding matrix.",
    )
    parser.add_argument(
        "--zero-init",
        action="store_true",
        help="Initialize the embedding matrix to zeros instead of random values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for embedding initialization.",
    )
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
    flat_vocab = flatten_vocab(vocab_data)
    validate_flat_vocab(flat_vocab)

    sorted_flat = {
        token: idx
        for token, idx in sorted(flat_vocab.items(), key=lambda item: item[1])
    }
    token_to_id_path = Path(args.out_flat)
    token_to_id_path.write_text(json.dumps(sorted_flat, indent=2), encoding="utf-8")

    id_to_token = {str(idx): token for token, idx in sorted_flat.items()}
    id_to_token_path = Path(args.out_ids)
    id_to_token_path.write_text(json.dumps(id_to_token, indent=2), encoding="utf-8")

    vocab_size = max(sorted_flat.values()) + 1
    embedding = build_embedding_matrix(
        vocab_size, hidden_dim=args.hidden_dim, zero_init=args.zero_init, seed=args.seed
    )
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required to save the embedding matrix. Install numpy and retry."
        ) from exc
    np.save(args.out_emb, embedding)

    print(f"Flattened vocabulary saved to: {token_to_id_path}")
    print(f"Id-to-token map saved to: {id_to_token_path}")
    print(f"Embedding matrix saved to: {args.out_emb} with shape {embedding.shape}")


if __name__ == "__main__":
    main()
