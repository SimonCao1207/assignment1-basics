from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = in_features - in_features.amax(dim=dim, keepdim=True)
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)


def cross_entropy_loss(logits: Float[Tensor, " bs V"], targets: Float[Tensor, " bs"]):
    """Computes the cross-entropy loss."""
    shifted: Float[Tensor, "bs V"] = logits - logits.max(dim=-1, keepdim=True).values  # for numerical stability
    log_z: Float[Tensor, " bs V"] = shifted.logsumexp(dim=-1)
    targets = rearrange(targets, "bs -> bs 1")
    target_logits = shifted.gather(dim=-1, index=targets)
    target_logits = rearrange(target_logits, "bs 1 -> bs")
    nll = -(target_logits - log_z)
    return nll.mean()


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
