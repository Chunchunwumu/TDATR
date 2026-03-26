""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

S3 (AutoFormerV2, https://arxiv.org/abs/2111.14725) Swin weights from
    - https://github.com/microsoft/Cream/tree/main/AutoFormerV2

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import logging
import math
from functools import partial
from typing import Optional, Callable, Tuple
import os
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_function
from timm.models.helpers import named_apply, checkpoint_seq
from timm.models.layers import to_2tuple, to_ntuple, trunc_normal_, _assert
from timm.models.vision_transformer import get_init_weights_vit

from TDATR.models.modules.layer import OutChannelParallelConv2D
from TDATR.models.modules.transformer_layer_effiency import ModelParallelMultiheadAttention,split_tensor
from TDATR_utils.utils import initialize_weight_cpu, initialize_weight_gpu
from TDATR_utils.models import MixedFusedLayerNorm as LayerNorm
from TDATR.models.modules.transformer_layer_effiency import attention_mask_func
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
from TDATR.modules.layer import ModelParallelMLP
from TDATR.modules.dense_attn import Window_CoreAttention

import torch.nn.functional as F
_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }




class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = OutChannelParallelConv2D(in_channels=in_chans, out_channels=embed_dim,
                                                  kernel_size=patch_size,  
                                                  stride=patch_size, 
                                                  gather_output=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x, bias = self.proj(x)
        if bias is not None:
            x = x + bias
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    dtype = x.dtype
    tmp_x = x.new_empty(shape).float()
    random_tensor = tmp_x.bernoulli_(keep_prob)
    random_tensor = random_tensor.to(dtype)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww



class MP_WindowAttention(ModelParallelMultiheadAttention):
    def __init__(
        self,
        layer_number: int,
        embed_dim: Optional[int]=None,
        num_heads: Optional[int]=None,
        kdim: Optional[int]=None,
        vdim: Optional[int]=None,
        bias: Optional[bool]=None,
        use_flash_attention: Optional[bool]=None,        
        self_attention: Optional[bool]=False,
        encoder_decoder_attention: Optional[bool]=False,
        attention_mask_func: Optional[Callable]=attention_mask_func,
        skip_last_bias_add: bool=True,
        qkv_bias: bool=False,
        window_size=7
    ):

        super(MP_WindowAttention, self).__init__(layer_number=layer_number, embed_dim=embed_dim, num_heads=num_heads,
                                                 kdim=kdim, vdim=vdim, bias=bias, use_flash_attention=use_flash_attention,        
                                                 self_attention=self_attention, encoder_decoder_attention=encoder_decoder_attention, 
                                                 attention_mask_func=attention_mask_func, skip_last_bias_add=skip_last_bias_add)
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), self.num_heads_per_partition, device=torch.cuda.current_device(), dtype=self.dtype))
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        initialize_weight_gpu(
            self.relative_position_bias_table,
            partial(normal_, std=.02),
            partition_dim=1,
            partition_stride=1
        )
        self.core_attention = Window_CoreAttention(
            head_dim=self.head_dim,
            num_head=self.num_heads_per_partition,
            max_seq_length=gpc.config.task.seq_length,
            position_type="none",
            attention_dropout_p=gpc.config.model.attention_dropout_p,
            use_fa_v2=True
        )
        
    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias#.unsqueeze(0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inference_params = None,
        **unused_kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attention_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attention_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        inference_key_memory, inference_value_memory = None, None
        local_size =  None if not gpc.config.generation.local_attention_memory_enable \
            else getattr(gpc.config.model, "sparse_local_size", None)
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = local_size if local_size != None else inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        if self.self_attention:
            # [len, batch, num_head * head_dim] -> [len, batch, num_head * 3 * head_dim]
            q_k_v, _ = self.q_k_v_proj(query)

            # [len, batch, num_head, 3 * head_dim] -> 3 * [len, batch, num_head * head_dim]
            q, k, v = split_tensor(q_k_v, num_partitions=3, dim=-1)
            q = q.transpose(0,1).contiguous()
            k = k.transpose(0,1).contiguous()
            v = v.transpose(0,1).contiguous()

        elif self.encoder_decoder_attention:
            q, _ = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.kdim == self.vdim:
                    k_v, _ = self.k_v_proj(key)
                    k, v = split_tensor(k_v, num_partitions=2, dim=-1)
                else:
                    k, _ = self.k_proj(key)
                    v, _ = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q, _ = self.q_proj(query)
            k, _ = self.k_proj(key)
            v, _ = self.v_proj(value)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + k.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + k.size(0)
            # Copy key and values.

            k_seqlen = k.size(0)
            batch_ids = inference_params.valid_batch_ids[batch_start:batch_end] if inference_params.valid_batch_ids != None \
                else torch.tensor([id for id in range(batch_start, batch_end)])
            if local_size!=None and sequence_end>local_size:
                if k_seqlen < local_size:
                    inference_key_memory[:, batch_ids, ...] = torch.roll(inference_key_memory[:, batch_ids, ...], -k_seqlen, 0)
                    inference_key_memory[local_size-k_seqlen:local_size, batch_ids, ...] = k
                    inference_value_memory[:, batch_ids, ...] = torch.roll(inference_value_memory[:, batch_ids, ...], -k_seqlen, 0)
                    inference_value_memory[local_size-k_seqlen:local_size, batch_ids, ...] = v
                    k = inference_key_memory[:local_size,  batch_ids, ...]
                    v = inference_value_memory[:local_size, batch_ids, ...]
                else:
                    inference_key_memory[:local_size, batch_ids, ...] = k[k_seqlen-local_size:k_seqlen,batch_ids,...]
                    inference_value_memory[:local_size, batch_ids, ...] = v[k_seqlen-local_size:k_seqlen,batch_ids,...]
            else:
                inference_key_memory[sequence_start:sequence_end,
                                    batch_ids, ...] = k
                inference_value_memory[sequence_start:sequence_end,
                                    batch_ids, ...] = v
                k = inference_key_memory[:sequence_end, batch_ids, ...]
                v = inference_value_memory[:sequence_end, batch_ids, ...]
        
        # =======================================
        # Compute attention scores.
        # =======================================
        relative_position_bias = self._get_rel_pos_bias()
        context = self.core_attention(
            q, 
            k, 
            v,
            padding_mask,
            attention_mask,
            relative_position_bias
        )

        # =================
        # Output. [s, b, h]
        # =================
        # context = context.transpose(0,1).contiguous()
        context = context.contiguous()
        attention_output, attention_bias = self.out_proj(context)

        return attention_output, attention_bias


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        new_swin_state_dict = {}
        for x in state_dict:
            if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                pass
            elif (
                x.endswith("relative_position_bias_table")
            ):
                pos_bias = state_dict[x].unsqueeze(0)[0]
                old_len = int(math.sqrt(len(pos_bias)))
                new_len = int(2 * self.window_size[0] - 1)
                if new_len == old_len:
                    new_swin_state_dict[x] = state_dict[x]
                    continue
                else:
                    print('Interpolate relative_position_bias_table !!!')
                pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                pos_bias = F.interpolate(pos_bias, size=(new_len, new_len), mode="bicubic", align_corners=False)
                new_swin_state_dict[x] = pos_bias.permute(0, 2, 3, 1).reshape(1, new_len ** 2, -1).squeeze(0)
            else:
                new_swin_state_dict[x] = state_dict[x]
        return super()._load_from_state_dict(new_swin_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, layer_num, dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = MP_WindowAttention(layer_number=layer_num, embed_dim=dim, num_heads=num_heads,
                                                kdim=dim, vdim=dim, self_attention=True, use_flash_attention=False, qkv_bias=qkv_bias, window_size=to_2tuple(self.window_size))
        # self.attn = nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ModelParallelMLP(dim, mlp_hidden_dim, dtype=self.dtype)

        self.hardtanh = torch.nn.Hardtanh(min_val=-32, max_val=32)

        # if self.shift_size > 0:
        #     # calculate attention mask for SW-MSA
        #     H, W = self.input_resolution
        #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        #     cnt = 0
        #     for h in (
        #             slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None)):
        #         for w in (
        #                 slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None)):
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1
        #     mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # else:
        #     attn_mask = None

        # self.register_buffer("attn_mask", attn_mask)

    def gen_mask(self, H, W, device="cpu"):
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        cnt = 0
        for h in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)):
            for w in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
        

    def forward(self, x):
        B,H, W,C=x.shape
        x = x.view(B,H*W,C)
        # _assert(L == H * W, "input feature has wrong size")

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # x = x.float()

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # shifted_x = shifted_x.to(self.dtype)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_mask = self.gen_mask(H, W, device=x.device) if self.shift_size > 0 else None
        attention_output, attention_bias = self.attn(x_windows, None, None, attention_mask=attn_mask)  # num_win*B, window_size*window_size, C
        attn_windows = attention_output+attention_bias
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # shifted_x = shifted_x.float()

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # x = x.to(self.dtype)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = self.hardtanh(x)

        mlp_output, mlp_bias = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_output+mlp_bias)

        x = x.view(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x, size):
        """
        x: B, H*W, C
        """
        H, W = size
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, (H//2, W//2)



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
            self, dim, out_dim, input_resolution, depth, num_heads=4, head_dim=None,
            window_size=7, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
            drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # self.grad_checkpointing = gpc.config.model.use_checkpoint

        self.grad_checkpointing = False #faster
        # self.grad_checkpointing = True 

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                layer_num=i ,dim=dim, input_resolution=input_resolution, num_heads=num_heads, head_dim=head_dim,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, size):
        
        h,w  = size
        b,l,c = x.shape
        x = x.view(b,h,w,c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = x.view(b,h*w,c)
        
        before_downsample = x
        H, W = size
        B, L, C = before_downsample.shape
        _assert(L == H * W, "input feature has wrong size")
        before_downsample = before_downsample.view(B, H, W, C)

        new_size = size
        if self.downsample is not None:
            x, new_size = self.downsample(x, new_size)
        return x, before_downsample, new_size



class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(
            self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), head_dim=None,
            window_size=7, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True, weight_init='', **kwargs):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features =[int(embed_dim * 2 ** (i)) for i in range(self.num_layers)]

        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
            
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        # import pdb; pdb.set_trace()

        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim, device=torch.cuda.current_device(), dtype=self.dtype)) if ape else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        layers = []
        self.norm_layers = nn.ModuleList()

        for i in range(self.num_layers):
            layers += [BasicLayer(
                dim=embed_dim[i],
                out_dim=embed_out_dim[i],
                input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                depth=depths[i],
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None
            )]
            self.norm_layers.append(norm_layer(self.num_features[i]))

        self.layers = nn.Sequential(*layers)

        ## FPN
        output_dim = 1024
        # self.conv_feat32 = nn.Conv2d(2048, output_dim, 1, 1, 0)
        # self.conv_feat16 = nn.Conv2d(1024, output_dim, 3, 1, 1)
        # self.conv_proj =  nn.Conv2d(output_dim, output_dim, 3, 1, 1)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        if self.absolute_pos_embed is not None:
            trunc_normal_(self.absolute_pos_embed, std=.02)
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.upsample(x, size=(H,W), mode='bilinear') + y
    
    def forward_features(self, x):
        B, C, H, W = x.shape
        

        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        patch_h, patch_w = self.patch_embed.patch_size
        feat_size = (H//patch_h, W//patch_w)
        

        feat_maps = []
        for layer_id, module in enumerate(self.layers):
            
            x, before_downsample, feat_size = module(x, feat_size)

            before_downsample = self.norm_layers[layer_id](before_downsample)
            feat_maps.append(before_downsample.permute(0,3,1,2))
        
        
        return feat_maps

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.forward_head(x)
        return x



def create_tp_swin_transformer(img_size=1024, patch_size=4, embed_dim=96, depths=(2, 2, 6, 2), 
                            num_heads=(3, 6, 12, 24), window_size=7, precision="bf16"):

    model = SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        norm_layer=nn.LayerNorm
    )  

    return model