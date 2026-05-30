import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        batch_size, seq_len, _ = x.size()

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            if attn_bias.shape != (batch_size, self.num_heads, seq_len, seq_len):
                raise ValueError(
                    f"attn_bias must be shape (batch, heads, seq, seq), got {attn_bias.shape}"
                )
            attn_scores = attn_scores + attn_bias

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        return self.out_proj(attn_output)


class TreeEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, parent_child_bias=None, key_padding_mask=None):
        residual = x
        attn_out = self.self_attn(
            x, attn_bias=parent_child_bias, key_padding_mask=key_padding_mask
        )
        x = self.norm1(residual + attn_out)
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + x)
        return x


class TreeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_heads=8,
        num_layers=8,
        ffn_dim=2048,
        dropout=0.1,
        position_dim=3,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_projection = nn.Linear(position_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TreeEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.parent_child_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, tokens, positions, parent_child_pairs, padding_mask=None):
        x = self.token_embedding(tokens) + self.position_projection(positions)

        batch_size, seq_len, _ = x.size()
        if parent_child_pairs is None:
            parent_child_pairs = torch.zeros(
                (batch_size, seq_len, seq_len), device=x.device, dtype=x.dtype
            )

        attn_bias = parent_child_pairs.unsqueeze(1) * self.parent_child_bias.view(
            1, -1, 1, 1
        )

        for layer in self.layers:
            x = layer(x, parent_child_bias=attn_bias, key_padding_mask=padding_mask)

        return self.norm(x)
