import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.utils import SDPA


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device, dtype=dtype), mean=0, std=2 / (d_in + d_out), a=-3, b=3
            )
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return x @ self.weight.t()


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
        # NOTE: d_ff should be set to approximately 8/3 × d_model
        # ensure that d_ff is a multiple of 64 to make good use of hardware.
        super().__init__()
        self.w1 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_in=d_ff, d_out=d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.silu = lambda x: x * F.sigmoid(x)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # SwiGLU
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        assert d_k % 2 == 0, "RoPE dimension must be even"
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        half_d = d_k // 2
        positions: Float[Tensor, " seq_len"] = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        positions = rearrange(positions, "seq_len -> seq_len 1")
        dim_index = torch.arange(half_d, device=device, dtype=torch.float32)
        dim_index = rearrange(dim_index, "half_d -> 1 half_d")
        angle_rates = 1.0 / (self.theta ** (dim_index / half_d))
        angles: Float[Tensor, "seq_len half_d"] = positions * angle_rates

        self.cos_cached: Float[Tensor, "max_seq_len half_d"]
        self.sin_cached: Float[Tensor, "max_seq_len half_d"]

        # Precompute the positional embeddings
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self, in_query_or_key: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        assert in_query_or_key.shape[-1] == self.d_k, "Input last dim must match d_k"

        cos: Float[Tensor, "... seq_len half_d"] = self.cos_cached[token_positions]
        sin: Float[Tensor, "... seq_len half_d"] = self.sin_cached[token_positions]
        x1 = in_query_or_key[..., ::2]  # even dim
        x2 = in_query_or_key[..., 1::2]  # odd dim

        out_even = x1 * cos - x2 * sin
        out_odd = x1 * sin + x2 * cos
        out: Float[Tensor, "... seq_len half_d 2"] = torch.stack((out_even, out_odd), dim=-1)
        out = rearrange(out, "... seq_len half_d two -> ... seq_len (half_d two)", two=2)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # NOTE: d_k and d_v are the same following Vaswani et al. (2017)
        self.d_k = self.d_v = d_model // n_heads

        self.q_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        rope: RoPE | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        Q = rearrange(self.q_proj(x), "... q (h d_k) -> ... h q d_k", h=self.n_heads)
        K = rearrange(self.k_proj(x), "... k (h d_k) -> ... h k d_k", h=self.n_heads)
        V = rearrange(self.v_proj(x), "... v (h d_v) -> ... h v d_v", h=self.n_heads)
        if rope is not None:
            if token_positions is None:
                seq_len = x.shape[-2]
                token_positions = torch.arange(0, seq_len, device=x.device, dtype=torch.int32)
            # Applying RoPE to Q and K but not V
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        q, k = Q.shape[-2], K.shape[-2]
        mask = torch.tril(torch.ones(q, k, device=x.device), diagonal=0)
        # broadcast to match the batch and head dimensions
        mask = rearrange(mask, "q k -> 1 1 q k")
        context: Float[Tensor, "... h s d_v"] = SDPA(query=Q, key=K, value=V, mask=mask)
        context = rearrange(context, "... h s d_v -> ... s (h d_v)")
        return self.output_proj(context)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, device=device, dtype=dtype)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta=rope_theta, d_k=self.attn.d_k, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        x = x + self.attn(self.ln1(x), token_positions=token_positions, rope=self.rope)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=num_heads,
                    d_ff=d_ff,
                    rope_theta=rope_theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        x = self.token_embeddings(token_ids)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)
