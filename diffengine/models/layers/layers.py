import torch
from torch import nn


def get_layer_norm(normalized_shape: int,
                   eps: float = 1e-5,
                   *,
                   elementwise_affine: bool = True) -> nn.Module:
    """Get the layer norm layer."""
    if torch.cuda.is_available():
        from .apex_layers import FusedLayerNorm
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)

    return nn.LayerNorm(normalized_shape, eps, elementwise_affine)
