import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
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


def get_lr_cosine_schedule(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    """Generates a learning rate based on a cosine schedule with warmup."""
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_lr + 0.5 * (max_lr - min_lr) * (
            1 + math.cos(torch.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        )
    else:
        return min_lr


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """
    Clips the gradients of the parameters to a maximum L2 norm.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norms = torch.norm(torch.stack([torch.norm(g) for g in grads]), p=2)
    if total_norms > max_l2_norm:
        clip_coef = max_l2_norm / (total_norms + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """
    x: integer 1-D numpy array with token IDs

    Returns the sampled input sequences and the corresponding next-token targets
    """
    assert x.ndim == 1, "x should be a 1-D numpy array"
    assert x.dtype == int, "x should contain integer token IDs"
    assert context_length > 0, "context_length must be positive"

    # Ensure the input is a PyTorch tensor
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)

    # Sample random starting indices for the batches
    start_indices = torch.randint(0, len(x_tensor) - context_length, (batch_size,), device=device)

    # Create input sequences and targets
    inputs = torch.stack([x_tensor[i : i + context_length] for i in start_indices])
    targets = torch.stack([x_tensor[i + 1 : i + 1 + context_length] for i in start_indices])

    return inputs, targets


def save_checkpoint(
    model: nn.Module, optimizer: Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        out,
    )


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: Optimizer) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


if __name__ == "__main__":
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    t1_c = tuple(nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    gradient_clipping(t1_c, max_l2_norm=1e-2)
