from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor
from torch.optim import Optimizer


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


class AdamW(Optimizer):
    """AdamW optimizer with weight decay."""

    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = grad.new_zeros(p.size())
                    state["exp_avg_sq"] = grad.new_zeros(p.size())

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * (bias_correction2**0.5 / bias_correction1)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])
