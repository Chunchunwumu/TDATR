# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from functools import partial


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))



def coordinate_to_encoding(coord_tensor: Tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """

    dim_t = torch.arange( num_feats, dtype=torch.float32, device=coord_tensor.device)

    # dim_t_1 = temperature**(2 * (dim_t // 2) / num_feats)
    dim_t=temperature**(2*torch.div(dim_t ,2, rounding_mode='floor')/num_feats)

    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    # if coord_tensor.size(-1) == 2:
    #     pos = torch.cat((pos_y, pos_x), dim=-1)
    # elif coord_tensor.size(-1) == 4:
    w_embed = coord_tensor[..., 2] * scale
    pos_w = w_embed[..., None] / dim_t
    pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                        dim=-1).flatten(2)

    h_embed = coord_tensor[..., 3] * scale
    pos_h = h_embed[..., None] / dim_t
    pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                        dim=-1).flatten(2)

    pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    # else:
    #     raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
    #         coord_tensor.size(-1)))
    return pos



def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x