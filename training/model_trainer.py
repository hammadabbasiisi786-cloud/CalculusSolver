import json
import os

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model.architecture import CalculusModel


class SLaNgTokenizer:
    def __init__(self):
        self.chars = sorted(
            list(
                " [](),.+-*/^0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
        )
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.char_to_idx["<PAD>"] = 0
        self.char_to_idx["<SOS>"] = len(self.char_to_idx)
        self.char_to_idx["<EOS>"] = len(self.char_to_idx)
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text, max_len=32):
        tokens = (
            [self.char_to_idx["<SOS>"]]
            + [self.char_to_idx.get(c, 0) for c in text]
            + [self.char_to_idx["<EOS>"]]
        )
        if len(tokens) < max_len:
            tokens += [self.char_to_idx["<PAD>"]] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, tokens):
        return "".join(
            [
                self.idx_to_char.get(t, "")
                for t in tokens
                if t
                not in [
                    self.char_to_idx["<PAD>"],
                    self.char_to_idx["<SOS>"],
                    self.char_to_idx["<EOS>"],
                ]
            ]
        )


class CalculusDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = torch.tensor(
            self.tokenizer.encode(item["input"], self.max_len), dtype=torch.long
        )
        tgt = torch.tensor(
            self.tokenizer.encode(item["output"], self.max_len), dtype=torch.long
        )

        src_positions = torch.zeros(self.max_len, 3, dtype=torch.float)
        parent_child_pairs = torch.zeros(self.max_len, self.max_len, dtype=torch.float)
        root_mask = torch.zeros(self.max_len, dtype=torch.bool)
        root_mask[0] = True

        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]

        return {
            "src": src,
            "src_positions": src_positions,
            "parent_child_pairs": parent_child_pairs,
            "root_mask": root_mask,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
            "rule_label": 0,
            "description_label": 0,
        }


def collate_batch(batch):
    return {key: torch.stack([sample[key] for sample in batch]) for key in batch[0]}


def train():
    data_path = os.path.join("data", "dataset.json")
    if not os.path.exists(data_path):
        print("Dataset not found. Please run data_generator.py first.")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    tokenizer = SLaNgTokenizer()
    train_data, val_data = train_test_split(data, test_size=0.1)

    train_dataset = CalculusDataset(train_data, tokenizer)
    val_dataset = CalculusDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CalculusModel(
        vocab_size=tokenizer.vocab_size,
        num_rules=None,
        hidden_dim=512,
        num_heads=8,
        num_layers=8,
        ffn_dim=2048,
        dropout=0.1,
        position_dim=3,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    decoder_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx["<PAD>"])
    rule_criterion = nn.CrossEntropyLoss()
    description_criterion = nn.CrossEntropyLoss()

    epochs = 100
    print(f"Starting training for {epochs} epochs on {device}...", flush=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            src = batch["src"].to(device)
            src_positions = batch["src_positions"].to(device)
            parent_child_pairs = batch["parent_child_pairs"].to(device)
            root_mask = batch["root_mask"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)
            rule_labels = batch["rule_label"].to(device)
            description_labels = batch["description_label"].to(device)

            decoder_logits, rule_logits, description_logits = model(
                src,
                src_positions,
                parent_child_pairs,
                tgt_input,
                root_mask=root_mask,
            )

            decoder_loss = decoder_criterion(
                decoder_logits.view(-1, tokenizer.vocab_size),
                tgt_output.reshape(-1),
            )
            rule_loss = rule_criterion(rule_logits, rule_labels)
            description_loss = description_criterion(
                description_logits, description_labels
            )

            loss = decoder_loss + rule_loss + 0.5 * description_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}",
            flush=True,
        )

    os.makedirs("model", exist_ok=True)
    model_data = {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer,
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 8,
            "ffn_dim": 2048,
        },
    }
    joblib.dump(model_data, os.path.join("model", "model.pkl"))
    print("Model saved as model/model.pkl", flush=True)


if __name__ == "__main__":
    train()
