from argparse import Namespace
from collections import OrderedDict
import re
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass, field
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
import functools
from fairscale.nn.checkpoint.checkpoint_utils import patch_batchnorm


from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.dataclass import HulkDataclass
from torch.nn import LayerNorm
from TDATR.models.modules.ophooks import BaseOpHook, register_ophooks
from TDATR_utils.utils import divide

from TDATR.models.modules.linear_layer import ColumnParallelLinear, RowParallelLinear
from TDATR_utils.utils import split_tensor

def bias_dropout_add(x, bias, residual, prob, training) :
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)

def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores

def _checkpoint_wrapper(
    module: nn.Module,
    offload_to_cpu: bool = False,
    distribute_checkpointed_activations: bool = False,
) -> nn.Module:
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:

        - wraps an nn.Module, so that all subsequent calls will use checkpointing
        - handles keyword arguments in the forward
        - handles non-Tensor outputs from the forward
        - supports offloading activations to CPU

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))

    To understand the benefits of checkpointing and the `offload_to_cpu` flag,
    let's divide activations into 2 types: inner activations and outer
    activations w.r.t. the checkpointed modules. The inner ones are saved
    by activation checkpointing, the outer ones are saved by offload_to_cpu.

    In terms of GPU memory savings:

        - When inner ones are large in size and outer ones are small,
          checkpointing helps a lot, offload_to_cpu may help a little.
        - When inner ones are small and outer ones are large,
          checkpointing helps little, offload_to_cpu helps a lot.
        - When both inner and outer are large, both help and the
          benefit is additive.

    ..Note::

        The first and last layers are not likely to benefit from the `offload_to_cpu` flag
        because (1) there are typically other references to the first layer's input, so
        the GPU memory won't be freed; (2) the input to the last layer is immediately
        used by the backward pass and won't result in memory savings.

    Args:
        module (nn.Module):
            The module to be wrapped
        offload_to_cpu (bool):
            Whether to offload activations to CPU.

    Returns:
        (nn.Module):
            Wrapped module
    """
    # Patch the batchnorm layers in case there are any in this module.
    patch_batchnorm(module)

    # The use of weakref here is to prevent creating a ref cycle: m -> m.forward -> m.
    # When such cycle exists, gc won't collect the module when the module is freed.
    # That causes GPU memory to be leaked. See the unit test for how we catch that.
    #
    # We prefer this over a class wrapper since the class wrapper would have to
    # proxy a lot of fields and methods.
    module.forward = functools.partial(  # type: ignore
        _checkpointed_forward,
        type(module).forward,
        weakref.ref(module),
        offload_to_cpu,
        distribute_checkpointed_activations,
    )
    return module

def checkpoint_wrapper(module, *args, **kwargs):
    module = _checkpoint_wrapper(module, *args, **kwargs)

    if hasattr(module, "extra_repr"):
        orig_extra_repr = module.extra_repr
    else:
        orig_extra_repr = None

    def extra_repr():
        return (
            f"[checkpointed] {orig_extra_repr()}" if orig_extra_repr is not None else ""
        )

    module.extra_repr = extra_repr

    return module





@dataclass
class TransformerEncoderLayerConfig(HulkDataclass):
    embed_dim: int = field(
        default=12, metadata={"help": "Hidden layer dimension"}
    )
    kdim: Optional[int] = field(
        default=None, metadata={"help": "Dimension of key layer"}
    )
    vdim: Optional[int] = field(
        default=None, metadata={"help": "Dimension of value layer"}
    )
    num_heads: int = field(
        default=32, metadata={"help": "Number of heads in MultiHeadAttention"}
    )
    num_kv_heads: Optional[int] = field(
        default=None,
        metadata={"help": "num kv heads in group query attention"},
    )
    attention_dropout_p: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout probability of core attention module"}
    )
    bias: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use bias"}
    )
    use_flash_attention: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use flash attention"}
    )
    self_attention: Optional[bool] = field(
        default=False, metadata={"help": "self_attention and encoder_decoder_attention are mutually exclusive"}
    )
    encoder_decoder_attention: Optional[bool] = field(
        default=False, metadata={"help": "self_attention and encoder_decoder_attention are mutually exclusive"}
    )
    attention_softmax_in_fp32: Optional[bool] = field(
        default=True, metadata={"help": "Whether Tensor is converted to float when calculating softmax"}
    )
    scaled_masked_softmax_fusion: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use fusion operator"}
    )
    mlp_embed_dim: int = field(
        default=12, metadata={"help": "The dimension of MLP layer and generally set to four times of embed_dim"}
    )
    hidden_dropout_p: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout probability of transformer module"}
    )
    layernorm_epsilon: Optional[float] = field(
        default=1e-5, metadata={"help": "Parameter configuration of layernorm"}
    )
    normalize_before: Optional[bool] = field(
        default=False, metadata={"help": "Set whether layernorm is in front or behind MLP and MHA"}
    )
    bias_dropout_fusion: Optional[bool] = field(
        default=False, metadata={"help": "Whether bias and dropout are fused"}
    )
    apply_residual_connection_post_layernorm: Optional[bool] = field(
        default=False, 
        metadata={"help": "Use original BERT residula connection ordering."}
    )
    apply_query_key_layer_scaling: Optional[bool] = field(
        default=True, metadata={"help": "Whether to scale the output of the layer"}
    )


class ModelParallelMultiheadAttention(nn.Module):
    def __init__(
        self,
        layer_number: int,
        embed_dim: Optional[int]=None,
        num_heads: Optional[int]=None,
        num_kv_heads: Optional[int]=None,
        kdim: Optional[int]=None,
        vdim: Optional[int]=None,
        bias: Optional[bool]=None,
        use_flash_attention: Optional[bool]=None,        
        self_attention: Optional[bool]=False,
        encoder_decoder_attention: Optional[bool]=False,
        attention_mask_func: Optional[Callable]=attention_mask_func,
        skip_last_bias_add: bool=True,
    ):
        """
            embed_dim: int,
            num_heads: int,
            kdim: Optional[int]=None,
            vdim: Optional[int]=None,
            dropout: Optional[float]=0.0,
            bias: Optional[bool]=True,
            layer_number: Optional[int]=None,
            apply_query_key_layer_scaling: Optional[bool]=False,
            self_attention: Optional[bool]=False,
            encoder_decoder_attention: Optional[bool]=False,
            attention_mask_func: Optional[Callable]=attention_mask_func,
            attention_softmax_in_fp32: Optional[bool]=True,
            scaled_masked_softmax_fusion: Optional[bool]=True,
            weight_init_method: Optional[Callable]=init.xavier_normal_,
            bias_init_method: Optional[Callable]=init.zeros_,
            dtype: Optional[torch.dtype]=torch.float16,
            use_cpu_initialization: Optional[bool]=True,
        """
        super().__init__()
        cfg = gpc.config.model

        self.embed_dim                     = embed_dim or cfg.embed_dim
        self.num_heads                     = num_heads or cfg.num_heads
        self.num_kv_heads                  = num_kv_heads or cfg.num_kv_heads or self.num_heads
        self.use_flash_attention           = use_flash_attention or cfg.use_flash_attention
        self.kdim                          = kdim or cfg.kdim or cfg.embed_dim
        self.vdim                          = vdim or cfg.vdim or cfg.embed_dim
        self.bias                          = bias or cfg.bias
        self.layer_number                  = layer_number
        self.self_attention                = self_attention
        self.encoder_decoder_attention     = encoder_decoder_attention
        self.attention_mask_func           = attention_mask_func

        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.use_cpu_initialization = gpc.config.model_parallel.use_cpu_initialization
        self.recompute_granularity = gpc.config.model_parallel.recompute_granularity
        self.model_parallel_size = gpc.get_world_size(ParallelMode.TENSOR)

        self.num_heads_per_partition = self.num_heads // self.model_parallel_size
        assert (
            self.num_heads_per_partition * self.model_parallel_size == self.num_heads
        ), "Number of heads must be divisible by model parallel size"

        self.num_kv_heads_per_partition = self.num_kv_heads // self.model_parallel_size
        assert (
            self.num_kv_heads_per_partition * self.model_parallel_size == self.num_kv_heads
        ), "Number of KV heads must be divisible by model parallel size"
        
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        self.embed_dim_per_partition = self.embed_dim // self.model_parallel_size
        assert (
            self.embed_dim_per_partition * self.model_parallel_size == self.embed_dim
        ), "embed_dim must be divisible by model_parallel_size"
        
        assert (
            not self.self_attention or  \
            ((self.kdim == self.embed_dim) and (self.vdim == self.embed_dim))
        ), "Self-attention requires query, key and value to be of the same size"

        if self.self_attention:
            if self.num_kv_heads != self.num_heads:                
                qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_kv_heads)
                self.q_k_v_proj = self.build_q_k_v_proj(
                    input_size=self.embed_dim,
                    output_size=qkv_dim,
                    bias=self.bias,
                    gather_output=False,
                    dtype=self.dtype,
                    use_cpu_initialization=self.use_cpu_initialization,
                    partition_stride=1,
                    skip_bias_add=False
                )
            else:
                self.q_k_v_proj = self.build_q_k_v_proj(
                    input_size=self.embed_dim,
                    output_size=self.embed_dim * 3,
                    bias=self.bias,
                    gather_output=False,
                    dtype=self.dtype,
                    use_cpu_initialization=self.use_cpu_initialization,
                    partition_stride=3,
                    skip_bias_add=False
                )
        elif self.encoder_decoder_attention:
            self.q_proj = self.build_q_proj(
                input_size=self.embed_dim,
                output_size=self.embed_dim,
                bias=self.bias,
                gather_output=False,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                skip_bias_add=False
            )
            if self.kdim == self.vdim:
                self.k_v_proj = self.build_k_v_proj(
                    input_size=self.kdim,
                    output_size=self.embed_dim,
                    bias=self.bias,
                    gather_output=False,
                    dtype=self.dtype,
                    use_cpu_initialization=self.use_cpu_initialization,
                    skip_bias_add=False
                )
            else:
                self.k_proj = self.build_k_proj(
                    input_size=self.kdim,
                    output_size=self.embed_dim,
                    bias=self.bias,
                    gather_output=False,
                    dtype=self.dtype,
                    use_cpu_initialization=self.use_cpu_initialization,
                    skip_bias_add=False
                )
                self.v_proj = self.build_v_proj(
                    input_size=self.vdim,
                    output_size=self.embed_dim,
                    bias=self.bias,
                    gather_output=False,
                    dtype=self.dtype,
                    use_cpu_initialization=self.use_cpu_initialization,
                    skip_bias_add=False
                )
        else:
            self.q_proj = self.build_q_proj(
                input_size=self.embed_dim,
                output_size=self.embed_dim,
                bias=self.bias,
                gather_output=False,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                skip_bias_add=False
            )
            self.k_proj = self.build_q_proj(
                input_size=self.kdim,
                output_size=self.embed_dim,
                bias=self.bias,
                gather_output=False,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                skip_bias_add=False
            )
            self.v_proj = self.build_q_proj(
                input_size=self.vdim,
                output_size=self.embed_dim,
                bias=self.bias,
                gather_output=False,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                skip_bias_add=False
            )
        
        self.core_attention = None


        if self.recompute_granularity == 'selective':
            self.core_attention = checkpoint_wrapper(self.core_attention)
   
        self.out_proj = self.build_out_proj(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=self.bias,
            dtype=self.dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            input_is_parallel=True,
            skip_bias_add=skip_last_bias_add,  # by default, skip bais add in last linear.
        )

    @staticmethod
    def build_q_k_v_proj(
        input_size: int,
        output_size: int,
        bias: Optional[bool]=True,
        weight_init_method: Optional[Callable]=init.xavier_normal_,
        bias_init_method: Optional[Callable]=init.zeros_,
        dtype: Optional[torch.dtype]=torch.float16,
        use_cpu_initialization: Optional[bool]=True,
        input_is_parallel: Optional[bool]=True,
        gather_output: Optional[bool]=True,
        partition_stride: Optional[int]=3,
        skip_bias_add: Optional[bool]=False,
    ):
        return ColumnParallelLinear(
            input_size,
            output_size, 
            bias=bias,
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            dtype=dtype,
            partition_stride=partition_stride,
            use_cpu_initialization=use_cpu_initialization
        )

    @staticmethod
    def build_k_v_proj(
        input_size: int,
        output_size: int,
        bias: Optional[bool]=True,
        weight_init_method: Optional[Callable]=init.xavier_normal_,
        bias_init_method: Optional[Callable]=init.zeros_,
        dtype: Optional[torch.dtype]=torch.float16,
        use_cpu_initialization: Optional[bool]=True,
        input_is_parallel: Optional[bool]=True,
        gather_output: Optional[bool]=True,
        skip_bias_add: Optional[bool]=False,
    ):
        return ColumnParallelLinear(
            input_size,
            output_size * 2, 
            bias=bias,
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            dtype=dtype,
            partition_stride=2,
            use_cpu_initialization=use_cpu_initialization
        )
    
    @staticmethod
    def build_q_proj(
        input_size: int,
        output_size: int,
        bias: Optional[bool]=True,
        weight_init_method: Optional[Callable]=init.xavier_normal_,
        bias_init_method: Optional[Callable]=init.zeros_,
        dtype: Optional[torch.dtype]=torch.float16,
        use_cpu_initialization: Optional[bool]=True,
        input_is_parallel: Optional[bool]=True,
        gather_output: Optional[bool]=True,
        skip_bias_add: Optional[bool]=False,
    ):
        return ColumnParallelLinear(
            input_size,
            output_size, 
            bias=bias,
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            dtype=dtype, 
            use_cpu_initialization=use_cpu_initialization
        )
    
    @staticmethod
    def build_k_proj(
        input_size: int,
        output_size: int,
        bias: Optional[bool]=True,
        weight_init_method: Optional[Callable]=init.xavier_normal_,
        bias_init_method: Optional[Callable]=init.zeros_,
        dtype: Optional[torch.dtype]=torch.float16,
        use_cpu_initialization: Optional[bool]=True,
        input_is_parallel: Optional[bool]=True,
        gather_output: Optional[bool]=True,
        skip_bias_add: Optional[bool]=False,
    ):
        return ColumnParallelLinear(
            input_size,
            output_size, 
            bias=bias,
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            dtype=dtype, 
            use_cpu_initialization=use_cpu_initialization
        )
    
    @staticmethod
    def build_v_proj(
        input_size: int,
        output_size: int,
        bias: Optional[bool]=True,
        weight_init_method: Optional[Callable]=init.xavier_normal_,
        bias_init_method: Optional[Callable]=init.zeros_,
        dtype: Optional[torch.dtype]=torch.float16,
        use_cpu_initialization: Optional[bool]=True,
        input_is_parallel: Optional[bool]=True,
        gather_output: Optional[bool]=True,
        skip_bias_add: Optional[bool]=False,
    ):
        return ColumnParallelLinear(
            input_size,
            output_size, 
            bias=bias,
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            dtype=dtype, 
            use_cpu_initialization=use_cpu_initialization
        )
    
    @staticmethod
    def build_out_proj(
        input_size: int,
        output_size: int,
        bias: Optional[bool]=True,
        weight_init_method: Optional[Callable]=init.xavier_normal_,
        bias_init_method: Optional[Callable]=init.zeros_,
        dtype: Optional[torch.dtype]=torch.float16,
        use_cpu_initialization: Optional[bool]=True,
        input_is_parallel: Optional[bool]=True,
        gather_output: Optional[bool]=True,
        skip_bias_add: Optional[bool]=False,
    ):
        return RowParallelLinear(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            input_is_parallel=input_is_parallel,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            dtype=dtype,
            use_cpu_initialization=use_cpu_initialization
        )
    
    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        seq_lengths: Optional[torch.LongTensor] = None,
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

        if inference_params is not None:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'
            if inference_params.__class__.__name__ == "InferenceParams":
                inference_params.create_kv_memory(
                    self.layer_number,
                    self.num_heads_per_partition * self.head_dim,
                    self.dtype
                )

        if self.self_attention:
            # [len, batch, num_head * head_dim] -> [len, batch, num_head * 3 * head_dim]
            q_k_v, _ = self.q_k_v_proj(query)
            if self.num_kv_heads is not None and self.num_kv_heads != self.num_heads:
                q = q_k_v[..., :self.num_heads_per_partition * self.head_dim]
                k, v = split_tensor(
                    q_k_v[..., self.num_heads_per_partition * self.head_dim:],
                    num_partitions=2,
                    dim=-1
                )
            else:
                # [len, batch, num_head, 3 * head_dim] -> 3 * [len, batch, num_head * head_dim]
                q, k, v = split_tensor(q_k_v, num_partitions=3, dim=-1)
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

        if inference_params is not None:
            inference_params.layer_idx = self.layer_number
            if inference_params.__class__.__name__ == "InferenceParams":
                k, v = inference_params.update_kv_memory(self.layer_number, k, v)

        # =======================================
        # Compute attention scores.
        # =======================================
        
        context = self.core_attention(
            q, 
            k, 
            v,
            padding_mask,
            attention_mask,
            seq_lengths,
            inference_params=inference_params,
        )

        # =================
        # Output. [s, b, h]
        # =================

        attention_output, attention_bias = self.out_proj(context)

        return attention_output, attention_bias


class ModelParallelMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_embed_dim: int,
        dtype:Optional[torch.dtype]=torch.float16,
        skip_last_bias_add: bool=True,
        partition_strides: Tuple[int, int]=(1, 1)
    ) -> None:
        super(ModelParallelMLP, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_embed_dim = mlp_embed_dim
        self.dtype = dtype
        self.use_cpu_initialization = gpc.config.model_parallel.use_cpu_initialization
        
        if partition_strides is not None:
            assert len(partition_strides) == 2, \
                f"Expected a Tuple[int, int], but got {partition_strides}"

        self.fc1 = self.build_fc1(
            gather_output=False,
            skip_bias_add=False,
            use_cpu_initialization=self.use_cpu_initialization,
            partition_stride=partition_strides[0],
        )
        self.activation_fn = self.build_activation()
        self.dropout = self.build_dropout(
            p=gpc.config.model.hidden_dropout_p,
        )
        self.fc2 = self.build_fc2(
            input_is_parallel=True,
            skip_bias_add=skip_last_bias_add,
            use_cpu_initialization=self.use_cpu_initialization,
            partition_stride=partition_strides[1],
        )
    
    def build_fc1(
        self, 
        gather_output: Optional[bool]=False,
        init_method: Optional[Callable]=init.xavier_normal_,
        skip_bias_add: Optional[bool]=False,
        use_cpu_initialization: Optional[bool]=True,
        partition_stride: int=1,
    ):
        return ColumnParallelLinear(
            self.embed_dim,
            self.mlp_embed_dim,
            gather_output=gather_output,
            dtype=self.dtype,
            weight_init_method=init_method,
            skip_bias_add=skip_bias_add,
            use_cpu_initialization=use_cpu_initialization,
            partition_stride=partition_stride
        )

    def build_fc2(
        self, 
        input_is_parallel: Optional[bool]=False,
        init_method: Optional[Callable]=init.xavier_normal_,
        skip_bias_add: Optional[bool]=False,
        use_cpu_initialization: Optional[bool]=True,
        partition_stride: int=1,
    ):
        return RowParallelLinear(
            self.mlp_embed_dim,
            self.embed_dim,
            input_is_parallel=input_is_parallel,
            dtype=self.dtype,
            weight_init_method=init_method,
            skip_bias_add=skip_bias_add,
            use_cpu_initialization=use_cpu_initialization,
            partition_stride=partition_stride
        )
    
    def build_activation(
        self,
    ):
        return nn.GELU()
    
    def build_dropout(
        self,
        p = 0.0
    ):
        return None
    
    def forward(self, inputs):
        intermediate_parallel, _ = self.fc1(inputs)
        intermediate_parallel = self.activation_fn(intermediate_parallel)
        output, output_bias = self.fc2(intermediate_parallel)
        
        return output, output_bias


class ModelParallelTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        layer_number: Optional[int]=None,
        self_attention: Optional[bool]=False,
        encoder_decoder_attention: Optional[bool]=False,
        attention_mask_func: Optional[Callable]=attention_mask_func,
    ):
        assert self_attention ^ encoder_decoder_attention, \
            "self_attention and encoder_decoder_attention cannot be both True or both False"
        super(ModelParallelTransformerEncoderLayer, self).__init__()
        self.layer_number = layer_number
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.attention_mask_func = attention_mask_func
        self.use_cpu_initialization = gpc.config.model_parallel.use_cpu_initialization
        if self.use_cpu_initialization:
            self.device = torch.device('cpu')
        else:
            self.device = torch.cuda.current_device()
        cfg = gpc.config.model
    
        self.embed_dim = cfg.embed_dim
        self.mlp_embed_dim = cfg.mlp_embed_dim
        self.num_heads = cfg.num_heads
        self.kdim = cfg.kdim
        self.vdim = cfg.vdim
        self.hidden_dropout_p = cfg.hidden_dropout_p
        self.normalize_before = cfg.normalize_before
        self.bias_dropout_fusion = cfg.bias_dropout_fusion
        self.apply_residual_connection_post_layernorm = cfg.apply_residual_connection_post_layernorm
        if not self.normalize_before and self.apply_residual_connection_post_layernorm:
            raise ValueError("apply_residual_connection_post_layernorm can only be false when \
                              normalize_before is false")
        self.sequence_parallel = gpc.config.model_parallel.sequence_parallel
        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.layer_norm = LayerNorm(
            self.embed_dim,
            eps=cfg.layernorm_epsilon,
            dtype=self.dtype,
            device=self.device,
        )

        skip_last_bias_add = True

        self.attention = self.build_attention(skip_last_bias_add)
        
        self.final_layer_norm = LayerNorm(
            self.embed_dim,
            eps=cfg.layernorm_epsilon,
            dtype=self.dtype,
            device=self.device,
        )

        self.mlp = self.build_mlp(skip_last_bias_add)


    def build_attention(
        self,
        skip_last_bias_add: bool=True,
    ):
        return ModelParallelMultiheadAttention(
            layer_number=self.layer_number,
            self_attention=self.self_attention,
            encoder_decoder_attention=self.encoder_decoder_attention,
            attention_mask_func=self.attention_mask_func,
            skip_last_bias_add=skip_last_bias_add,
        )

    def build_mlp(
        self,
        skip_last_bias_add: bool=True,
    ):
        return ModelParallelMLP(
            embed_dim=self.embed_dim,
            mlp_embed_dim=self.mlp_embed_dim,
            dtype=self.dtype,
            skip_last_bias_add=skip_last_bias_add,
        )

    def forward(
        self, 
        hidden_states,
        padding_mask,
        query_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attention_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attention_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # Stage 1: MultiHeadAttention
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)
            if self.apply_residual_connection_post_layernorm:
                residual = hidden_states
        
        if gpc.config.common.fp32_residual_connection:
            hidden_states = hidden_states.to(self.dtype)
        
        if query_hidden_states is None:
            query_hidden_states = hidden_states

        attention_output, attention_bias = self.attention(
            query=query_hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=padding_mask,
            attention_mask=attention_mask,
            seq_lengths=seq_lengths,
            inference_params=inference_params
        )

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                fused_bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                fused_bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            fused_bias_dropout_add_func = get_bias_dropout_add(self.training)

        # (NOTE) if sequence parallel enabled, should use different rng
        # state for `dropout` cross sequence(tensor) parallel region.
        rng_context = nullcontext()
        with torch.enable_grad(), rng_context:
            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)
            hidden_states = fused_bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout_p
            )

        if not self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)

        # Stage 2: FeedForward
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
            if self.apply_residual_connection_post_layernorm:
                residual = hidden_states

        if gpc.config.common.fp32_residual_connection:
            hidden_states = hidden_states.to(self.dtype)
        
        mlp_output, mlp_bias = self.mlp(hidden_states)

        # (NOTE) if sequence parallel enabled, should use different rng
        # state for `dropout` cross sequence(tensor) parallel region.
        rng_context = nullcontext()
        with torch.enable_grad(), rng_context:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            hidden_states = fused_bias_dropout_add_func(
                mlp_output,
                mlp_bias,
                residual,
                self.hidden_dropout_p
            )
        
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

          
def set_pipeline_parallel_attrs(module: nn.Module, num_layers: int, offset: int) -> None:
    pp_rank  = gpc.get_local_rank(ParallelMode.PIPELINE)
    pp_size  = gpc.get_world_size(ParallelMode.PIPELINE)
    vpp_rank = gpc.virtual_pipeline_parallel_rank
    vpp_size = gpc.virtual_pipeline_parallel_size
    attr = {
        "pp_size": pp_size,
        "pp_rank": pp_rank,
        "vpp_size": int(vpp_size) if vpp_size is not None else -1,
        "vpp_rank": int(vpp_rank) if vpp_rank is not None else -1,
        "num_layers": num_layers,
        "offset": offset,
    }
    setattr(module, "pipeline_model_parallel", True)
    setattr(module, "pipeline_parallel_attrs", attr)


class PipelineParallelBaseModel(nn.Module):
    def __init__(self) -> None:
        super(PipelineParallelBaseModel, self).__init__()
        self.pre_process = gpc.is_pipeline_first_stage()
        self.post_process = gpc.is_pipeline_last_stage()
        self.input_tensor = None

    def build_layer(self,
                    cfgs: Namespace,
                    *args, **kwargs) -> nn.Module:
        """Build single layer."""
        raise NotImplementedError

    def build_layers(self,
                     cfgs: Namespace,
                     total_layers: int,
                     *args, **kwargs) -> Tuple[int, nn.ModuleList]:
        """Calculate num_layers and build layers for pipeline parallelism."""
        mpu_cfg = gpc.config.model_parallel
  
        pp_rank  = gpc.get_local_rank(ParallelMode.PIPELINE)
        pp_size  = gpc.get_world_size(ParallelMode.PIPELINE)
        vpp_rank = gpc.virtual_pipeline_parallel_rank
        vpp_size = gpc.virtual_pipeline_parallel_size

        assert total_layers % pp_size == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = total_layers // pp_size

        if vpp_size is not None and vpp_rank is not None:
            assert total_layers % vpp_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // vpp_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = vpp_rank * (total_layers // vpp_size) + (pp_rank * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = pp_rank * self.num_layers

        layers = []
        for i in range(self.num_layers):
            kwargs.update(global_index=i + offset)
            layer = self.build_layer(cfgs, *args, **kwargs)
            setattr(layer, "offset", offset)
            setattr(layer, "global_index", i + offset)
            if mpu_cfg.recompute_granularity == 'full':
                layer = checkpoint_wrapper(
                    layer,
                    mpu_cfg.offload_activations,
                    mpu_cfg.distribute_checkpointed_activations,
                )
            layers.append(layer)

        self.layers = torch.nn.ModuleList(layers)
        self._register_layers_state_dict_hooks(self.layers, offset)
        set_pipeline_parallel_attrs(self.layers, total_layers, offset)
        return self.num_layers, self.layers

    def _get_layer(self, layer_index):
        """Get layer in current pipeline stage"""
        return self.layers[layer_index]

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    @staticmethod
    def _register_layers_state_dict_hooks(layers: nn.Module, offset: int):
        layers._register_state_dict_hook(
            functools.partial(_layers_state_dict_hook, offset=offset))
        layers._register_load_state_dict_pre_hook(
            functools.partial(_layers_load_state_dict_pre_hook, offset=offset))


def _layers_state_dict_hook(module: nn.Module,
                            state_dict: OrderedDict,
                            prefix: str,
                            local_metadata: OrderedDict,
                            offset: int):
    org_state_keys = list(state_dict)
    pattern = f"^{prefix}([0-9]+)\."
    for k in org_state_keys:
        value = state_dict[k]
        search_result = re.search(pattern, k)
        if search_result is not None:
            local_idx = int(search_result.group(1))
            new_k = re.sub(pattern, f"{prefix}{local_idx + offset}.", k, 1)
            state_dict.pop(k)
            state_dict[new_k] = value


def _layers_load_state_dict_pre_hook(state_dict: OrderedDict,
                                     prefix: str,
                                     local_metadata: OrderedDict,
                                     strict: str,
                                     missing_keys: List[str],
                                     unexpected_keys: List[str],
                                     error_msgs: List[str],
                                     offset: int):

    org_state_keys = list(state_dict)
    pattern = f"^{prefix}([0-9]+)\."
    for k in org_state_keys:
        value = state_dict[k]
        search_result = re.search(pattern, k)
        if search_result is not None:
            global_idx = int(search_result.group(1))
            new_k = re.sub(pattern, f"{prefix}{global_idx - offset}.", k, 1)
            state_dict.pop(k)
            state_dict[new_k] = value