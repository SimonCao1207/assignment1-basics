from einops import einsum
from jaxtyping import Float
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = in_features - in_features.amax(dim=dim, keepdim=True)
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)


def SDPA(
    query: Float[Tensor, "... q d_k"],
    key: Float[Tensor, "... k d_k"],
    value: Float[Tensor, "... k d_v"],
    mask: Float[Tensor, "... q k"] | None = None,
) -> Float[Tensor, "... q d_v"]:
    """Scaled Dot Product Attention function."""
    d_k = query.shape[-1]
    scale = d_k**-0.5
    scores = einsum(query, key, "... q d_k, ... k d_k -> ... q k") * scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    weighted_value = einsum(attn_weights, value, "... q k, ... k d_v -> ... q d_v")
    return weighted_value
