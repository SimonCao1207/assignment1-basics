import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device, dtype=dtype), mean=0, std=2 / (d_in + d_out), a=-3, b=3
            )
        )
        # NOTE: bias can be removed in modern LLM
        self.bias = nn.Parameter(torch.zeros(d_out, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return x @ self.weight.t() + self.bias


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


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight: Float[Tensor, " d_model"] = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        # Upcast to prevent overflow
        x = x.to(torch.float32)
        result = x * (self.weight / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
        return result.to(in_dtype)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_in=d_ff, d_out=d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.silu = lambda x: x * torch.nn.functional.sigmoid(x)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # SwiGLU
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
