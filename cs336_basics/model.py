import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weights: Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device, dtype=dtype), mean=0, std=2 / (d_in + d_out), a=-3, b=3
            )
        )
        self.bias = nn.Parameter(torch.zeros(d_out, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return x @ self.weights.t() + self.bias


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Float[Tensor, "vocab_size d_model"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), mean=0, std=1
            )
        )

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]
