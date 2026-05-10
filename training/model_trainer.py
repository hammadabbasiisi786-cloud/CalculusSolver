import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Simple character-level tokenizer for SLaNg
class SLaNgTokenizer:
    def __init__(self):
        self.chars = sorted(list(" [](),.+-*/^0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<SOS>'] = len(self.char_to_idx)
        self.char_to_idx['<EOS>'] = len(self.char_to_idx)
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text, max_len=32):
        tokens = [self.char_to_idx['<SOS>']] + [self.char_to_idx.get(c, 0) for c in text] + [self.char_to_idx['<EOS>']]
        if len(tokens) < max_len:
            tokens += [self.char_to_idx['<PAD>']] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, tokens):
        return "".join([self.idx_to_char.get(t, "") for t in tokens if t not in [self.char_to_idx['<PAD>'], self.char_to_idx['<SOS>'], self.char_to_idx['<EOS>']]])

class CalculusDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = torch.tensor(self.tokenizer.encode(item['input'], self.max_len))
        tgt = torch.tensor(self.tokenizer.encode(item['output'], self.max_len))
        return src, tgt

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        
        # Create masks
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(output)

def train():
    # Load data
    data_path = "CalculusSolver/data/dataset.json"
    if not os.path.exists(data_path):
        print("Dataset not found. Please run data_generator.py first.")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    tokenizer = SLaNgTokenizer()
    train_data, val_data = train_test_split(data, test_size=0.1)
    
    train_dataset = CalculusDataset(train_data, tokenizer)
    val_dataset = CalculusDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])

    epochs = 100
    print(f"Starting training for {epochs} epochs on {device}...", flush=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Input to transformer: tgt shifted by 1
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), tgt_expected.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}", flush=True)

    # Save model and tokenizer
    os.makedirs("CalculusSolver/model", exist_ok=True)
    model_data = {
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4
        }
    }
    joblib.dump(model_data, "CalculusSolver/model/model.pkl")
    print("Model saved as model.pkl", flush=True)

if __name__ == "__main__":
    train()
