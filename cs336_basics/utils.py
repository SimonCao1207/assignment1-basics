from jaxtyping import Float
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = in_features - in_features.amax(dim=dim, keepdim=True)
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)
