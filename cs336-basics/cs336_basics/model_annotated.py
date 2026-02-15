import torch
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum
import math

import torch.cuda.nvtx as nvtx

from .nn_utils import softmax

# NVTX (NVIDIA Tools Extension) is an API you use to add markers and ranges to your code so profilers like Nsight Systems show labeled sections in the timeline. It doesn’t change performance—just annotations for profiling.
# Notes: @nvtx.range('scaled dot product attention') is a decorator that wraps a function in an NVTX range named “scaled dot product attention.” When you profile with Nsight Systems, you’ll see a labeled region with that name in the timeline around that function’s execution. It’s just for profiling/visualization, not logic.
# with nvtx.range('computing softmax'): a context manager. Similar effects.

@nvtx.range('scaled dot product attention')
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]

    with nvtx.range('computing the attention scores from Q and K'):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    with nvtx.range('adding masks'):
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range('computing softmax'):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range('computing the final output'):
        out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return out