import torch
import torch.nn as nn
from jaxtyping import Float
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
