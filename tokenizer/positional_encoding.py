import torch
import torch.nn as nn
from typing import Iterable, List, Union


def _normalize_path_bit(decision: Union[int, str, bool]) -> int:
    if isinstance(decision, bool):
        return 1 if decision else 0
    if isinstance(decision, str):
        return 1 if decision.lower() in {"r", "right", "1", "true"} else 0
    return 1 if int(decision) != 0 else 0


def path_hash_scalar(path: Iterable[Union[int, str, bool]], base: int = 3) -> float:
    """Deterministically hash a left/right branch decision path into a scalar."""
    h = 1
    for decision in path:
        h = h * base + _normalize_path_bit(decision)
    return float(h)


def path_hashes(
    paths: List[Iterable[Union[int, str, bool]]], base: int = 3
) -> torch.Tensor:
    """Convert a batch of left/right paths into a numeric tensor for projection."""
    values = [path_hash_scalar(path, base=base) for path in paths]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


class TreePositionEncoding(nn.Module):
    """Three-signal tree position embedding for transformer inputs.

    The encoder input embedding is augmented by:
      1) depth projection
      2) sibling index projection
      3) deterministic path hash projection
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        if hidden_dim < 3:
            raise ValueError("hidden_dim must be at least 3.")
        self.hidden_dim = hidden_dim
        self._third = hidden_dim // 3
        self._remainder = hidden_dim - 2 * self._third
        self.depth_proj = nn.Linear(1, self._third)
        self.sibling_proj = nn.Linear(1, self._third)
        self.path_proj = nn.Linear(1, self._remainder)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        path_hashes: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.dim() != 3:
            raise ValueError(
                "token_embeddings must be a 3D tensor [batch, seq_len, hidden_dim]."
            )

        batch_size, seq_len, hidden_dim = token_embeddings.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Embedding dimension {hidden_dim} does not match hidden_dim {self.hidden_dim}."
            )

        def _prepare(feature: torch.Tensor, name: str) -> torch.Tensor:
            if feature.dim() == 1:
                feature = feature.unsqueeze(-1)
            if feature.dim() != 2:
                raise ValueError(f"{name} must be a 1D or 2D tensor.")
            if feature.shape[0] != batch_size or feature.shape[1] != seq_len:
                raise ValueError(f"{name} shape must match [batch, seq_len].")
            return feature.to(dtype=token_embeddings.dtype)

        depth_feat = _prepare(depths, "depths")
        sibling_feat = _prepare(sibling_indices, "sibling_indices")
        path_feat = _prepare(path_hashes, "path_hashes")

        depth_embed = self.depth_proj(depth_feat)
        sibling_embed = self.sibling_proj(sibling_feat)
        path_embed = self.path_proj(path_feat)

        position_embed = torch.cat([depth_embed, sibling_embed, path_embed], dim=-1)
        return token_embeddings + position_embed

    def encode_paths(
        self,
        paths: List[Iterable[Union[int, str, bool]]],
        device: Union[torch.device, str, None] = None,
    ) -> torch.Tensor:
        tensor = path_hashes(paths).to(
            device if device is not None else self.depth_proj.weight.device
        )
        return tensor


if __name__ == "__main__":
    encoding = TreePositionEncoding(hidden_dim=128)
    tokens = torch.zeros((2, 5, 128), dtype=torch.float32)
    depths = torch.tensor([[0, 1, 2, 2, 1], [0, 1, 1, 2, 2]], dtype=torch.float32)
    siblings = torch.tensor([[0, 0, 1, 2, 0], [0, 0, 1, 0, 1]], dtype=torch.float32)
    paths = [[0, 0, 1], [0, 1, 0], [0, 0], [0, 1, 1], [0]]
    path_hashes_tensor = encoding.encode_paths(paths)
    output = encoding(tokens, depths, siblings, path_hashes_tensor)
    print("TreePositionEncoding output shape:", output.shape)
