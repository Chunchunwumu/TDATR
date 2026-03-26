from contextlib import nullcontext
import os
from typing import Optional
from functools import partial
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from TDATR.models.modules.base_model import BaseHulkModel


from TDATR.models.modules.transformer_layer_effiency import bias_dropout_add_fused_inference, bias_dropout_add_fused_train, checkpoint_wrapper, get_bias_dropout_add
from TDATR_utils.forward_step import InferenceParams
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
from torch.nn import LayerNorm as LayerNorm
from TDATR_utils.models import parallel_lm_logits

from TDATR.models.modules.transformer_layer_effiency import attention_mask_func, TransformerEncoderLayerConfig,ModelParallelMLP
from TDATR.models.modules.transformer_layer_effiency import PipelineParallelBaseModel


from TDATR.modules.attention import EmbeddingEx_cfgi
from TDATR.modules.xpos import RotaryPositionalTransform_row_col
from typing import Optional, Callable, Tuple

from TDATR.models.modules.linear_layer import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from TDATR.models.modules.transformer_layer_effiency import split_tensor


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

        seq_parallel_size = gpc.get_world_size(ParallelMode.SEQ)
        seq_parallel_algo = gpc.config.model_parallel.sequence_parallel_algo
        

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
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params: Optional[InferenceParams] = None,
        **unused_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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



class ModelParallelTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        layer_number: Optional[int]=None,
        self_attention: Optional[bool]=False,
        encoder_decoder_attention: Optional[bool]=False,
        attention_mask_func: Optional[Callable]=attention_mask_func,
        cfg=None,
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
        # cfg = gpc.config.model
    
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
        self.ROPE_rc = RotaryPositionalTransform_row_col(self.embed_dim)
        skip_last_bias_add = True

        self.row_attention = self.build_attention(skip_last_bias_add)
        self.col_attention = self.build_attention(skip_last_bias_add)
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
        query_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params = None,
        row_col_positions = None,
        row_same_mask = None,
        col_same_mask = None,
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
            row_col_positions: (L,B,4) for ROPE, later will add relative bias
            row_same_mask: (L,L),
            col_same_mask: (L,L),

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # 修改改成三个： 行列的位置编码，利用一维rope，并且加上（起始的行列序号
        # 整体的编码，不加位置编码
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
        query_hidden_states = query_hidden_states
        hidden_states, attention_bias = self.row_attention(
            query=query_hidden_states,
            key=query_hidden_states,
            value=hidden_states,
            key_padding_mask=padding_mask,
            attention_mask=row_same_mask,
            seq_lengths=seq_lengths,
            inference_params=inference_params
        )
        query_hidden_states = hidden_states
        hidden_states, attention_bias = self.col_attention(
            query=query_hidden_states,
            key=query_hidden_states,
            value=hidden_states,
            key_padding_mask=padding_mask,
            attention_mask=col_same_mask,
            seq_lengths=seq_lengths,
            inference_params=inference_params
        )
        attention_output, attention_bias = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=padding_mask,
            attention_mask=None,
            seq_lengths=seq_lengths,
            inference_params=inference_params
        )

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        fused_bias_dropout_add_func = bias_dropout_add_fused_inference
        
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


@dataclass
class IPTConfig(TransformerEncoderLayerConfig):
    num_layers: Optional[int] = field(
        default=31,
        metadata={
            "help": ""
        },
    )
    padded_vocab_size: Optional[int] = field(
        default=40000,
        metadata={
            "help": ""
        },
    )
    max_position_embeddings: Optional[int] = field(
        default=4096,
        metadata={
            "help": ""
        },
    )
    init_method_std: Optional[float] = field(
        default=0.02,
        metadata={
            "help": ""
        },
    )
    from_pretrained: Optional[str] = field(
        default=None,
        metadata={
            "help": "Load pretrained pangu-alpha model if from_pretrained is not None"
        },
    )
    parallel_output: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether the model output result is parallel. " \
            "Suggestions: " \
            "(1) It is better to set it to True in the training stage; " \
            "(2) It is better to set it to False in the inference stage;"
        },
    )
    cross_flash_attn: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether using flash attn in cross attn. " \
        },
    )


class Embedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        dtype,
        init_method,
        num_tokentypes=0
    ):
        super(Embedding, self).__init__()

        self.embed_dim = embed_dim
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.sequence_parallel = gpc.config.model_parallel.sequence_parallel
        self.use_cpu_initialization = gpc.config.model_parallel.use_cpu_initialization
        self.fp32_residual_connection = gpc.config.common.fp32_residual_connection
        
        if self.use_cpu_initialization:
            device = torch.device("cpu")
        else:
            device = torch.cuda.current_device()

        # Word embeddings (parallel).
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, 
            self.embed_dim,
            None,
            dtype=dtype,
            init_method=self.init_method,
            use_cpu_initialization=self.use_cpu_initialization
        )

        # Position embedding (serial).
        # self.position_embeddings = torch.nn.Embedding(
        #     num_embeddings = max_sequence_length,
        #     embedding_dim = self.embed_dim,
        #     dtype=dtype,
        #     device=device
        # )
        # self.init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def forward(self, tokens, position_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(tokens)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        embeddings = embeddings.transpose(0, 1).contiguous()

        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings




class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

#--------------------MultiHeadAttention-----------------------------
try:
    from flash_attn import flash_attn_varlen_kvpacked_func
except:
    flash_attn_varlen_kvpacked_func=None

import math
from einops import rearrange
class FlashCrossAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        dtype: torch.dtype = torch.float16,
        attention_dropout_p: float = 0.0,
        causal: bool = True,
        **kwargs
    ) -> None:
        super(FlashCrossAttention, self).__init__()
        assert flash_attn_varlen_kvpacked_func is not None or gpc.config.common.npu, \
            f"import flash attention error on gpu: {import_error_msg}"
        self.head_dim = head_dim
        self.num_head = num_head
        
        self.dtype=dtype

        self.norm_factor = 1. / math.sqrt(head_dim)
        self.dropout_p = attention_dropout_p
        self.causal = causal

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lengths: Optional[torch.LongTensor] = None
    ):
        if q.device.type == 'cuda':
            output = self.gpu_flash(q, k, v)
        elif q.device.type == 'npu':
            q = q.transpose(0, 1).contiguous()
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()
            output = self.npu_flash(q, k, v).transpose(0, 1).contiguous()
        return output

    def gpu_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lengths: Optional[torch.LongTensor] = None
    ):
        bsz, hidden_size = q.shape[1], q.shape[2]
        seqlen_q, seqlen_k = q.shape[0], k.shape[0]
        assert hidden_size == self.num_head * self.head_dim
        assert seq_lengths is None, \
            "`FlashCrossAttention` dosen't support `seq_lengths`"

        q = rearrange(q, 's b (h d) -> (b s) h d', h = self.num_head, d = self.head_dim)
        k = rearrange(k, 's b (h d) -> (b s) h d', h = self.num_head, d = self.head_dim)
        v = rearrange(v, 's b (h d) -> (b s) h d', h = self.num_head, d = self.head_dim)
        kv = torch.stack([k, v], dim=1)     # kv's shape: (b s) 2 h d
        
        cu_seqlens_q = torch.arange(
            0,
            (bsz + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q.device
        )
        cu_seqlens_k = torch.arange(
            0,
            (bsz + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=q.device
        )
        output = flash_attn_varlen_kvpacked_func(
            q=q,
            kv=kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.norm_factor,
            causal=self.causal
        )

        output = rearrange(output, '(b s) ... -> s b (...)', b=bsz)
        
        return output

    def npu_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        B, N, _ = q.shape
        _, S, _ = k.shape
        q = rearrange(q, "b s (h d) -> (b s) h d", h=self.num_head, d=self.head_dim)
        k = rearrange(k, "b s (h d) -> (b s) h d", h=self.num_head, d=self.head_dim)
        v = rearrange(v, "b s (h d) -> (b s) h d", h=self.num_head, d=self.head_dim)
        # logger.info(f"TND: q.shape {q.shape}, k.shape {k.shape}, v.shape {v.shape}")

        actual_seq_qlen = []
        ans = 0
        for _ in range(B):
            ans += N
            actual_seq_qlen.append(ans)
        if padding_mask is None:
            padding_mask = []
            ans = 0
            for _ in range(B):
                ans += S
                padding_mask.append(ans)

        output = torch_npu.npu_fusion_attention(
            q, k, v, self.num_head, "TND",
            pse=None,
            # pre_tockens=2147483647,
            # next_tockens=2147483647,
            actual_seq_qlen=tuple(actual_seq_qlen),
            actual_seq_kvlen=tuple(padding_mask),
            scale=self.norm_factor,
            keep_prob=1.-self.dropout_p,
            sparse_mode=0)[0]
        
        output = rearrange(output.contiguous(), "(b s) h d -> b s (h d)", b=B, s=N)
        return output
    

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # self.W_Q = nn.Linear(input_dim, input_dim)
        # self.W_K = nn.Linear(input_dim, input_dim)
        # self.W_V = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, embed_dim)

        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.flash_cross_attention = FlashCrossAttention(head_dim=self.head_dim, 
                                num_head=self.num_heads,
                                dtype = self.dtype,
                                attention_dropout_p=dropout,
                                causal=False
                                )
        
    def split_heads(self, x, batch_size):
        assert x.shape[-1] % self.num_heads == 0, "Input dimension must be divisible by the number of heads"
        # Reshape input to (B, L, C) -> (batch_size, num_heads, seq_len, head_dim)
        x = x.view(batch_size, -1, self.num_heads, x.shape[-1] // self.num_heads)
        return x.permute(0, 2, 1, 3)
    
    def naiive_attn(self, query, key, value, attn_mask=None, INF = 100000):
        query = query.permute(1, 0, 2) #L,B,C-> B,L,C
        key = key.permute(1, 0, 2) #L,B,C-> B,L,C
        value = value.permute(1, 0, 2) #L,B,C-> B,L,C
        # _logger.info(f"query:{query.shape}, key:{key.shape}, value:{value.shape}")
        batch_size = query.size(0)
        
        # Linear transformation for each component
        # Q = self.W_Q(query)
        # K = self.W_K(key)
        # V = self.W_V(value)
        Q, K, V = query, key, value
        
        # Split the heads
        Q = self.split_heads(Q, batch_size).float()
        K = self.split_heads(K, batch_size).float()
        V = self.split_heads(V, batch_size)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if attn_mask is not None:
            # attention_scores += attn_mask
            if self.training:
                attention_scores = attention_scores.masked_fill(attn_mask == False, -INF)

        
        # _logger.info(f'Q.shape:{Q.shape} K.shape:{K.shape} attention_scores.shape:{attention_scores.shape}')
        attention_scores = F.softmax(attention_scores, dim=-1).to(dtype=V.dtype)
        attention_weights = self.dropout(attention_scores)
        
        context = torch.matmul(attention_weights, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.input_dim) #B,L,C
        return context

    def forward(self, query, key, value, attn_mask=None):
        if self.training:
            if gpc.config.model.cross_flash_attn:
                context = self.flash_cross_attention(query, key, value).transpose(0,1).contiguous()
                # print('FA')
            else:
                context = self.naiive_attn(query, key, value, attn_mask)
            # Final linear transformation
            output = self.fc(context)
            return output, None
        else:
            query = query.permute(1, 0, 2) #L,B,C-> B,L,C
            key = key.permute(1, 0, 2) #L,B,C-> B,L,C
            value = value.permute(1, 0, 2) #L,B,C-> B,L,C

            batch_size = query.size(0)
            
            # Linear transformation for each component
            # Q = self.W_Q(query)
            # K = self.W_K(key)
            # V = self.W_V(value)
            Q, K, V = query, key, value
            
            # Split the heads
            Q = self.split_heads(Q, batch_size)
            K = self.split_heads(K, batch_size)
            V = self.split_heads(V, batch_size)
            
            # Compute scaled dot-product attention
            attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
            
            if attn_mask is not None:
                attention_scores += attn_mask
            
            attention_scores = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_scores)

            context = torch.matmul(attention_weights, V)
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(batch_size, -1, self.input_dim) #B,L,C
            
            # Final linear transformation
            output = self.fc(context)
            return output, None




class CrossAtten_FFN(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            kv_dim=None,
            **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = FeedForwardNetwork(embed_dim, embed_dim*2, embed_dim, dropout=0.1)
        # self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.hardtanh = nn.Hardtanh(min_val=-32.0, max_val=32.0)

        self.attention = MultiheadAttention(embed_dim, embed_dim, num_heads)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, query, kv, attn_mask=None):
        ## q [L. B C]
        # print(f'kv shape: {kv.shape}')
        que = self.proj_q(query)
        # key = self.proj_k(kv)
        out = self.attention(
            que, 
            kv, 
            kv,
            attn_mask=attn_mask)[0]  # nn.MultiheadAttention:[L B C]  MultiheadAttention:[B  L C] 
        # out = out.permute(1, 0, 2) # nn.MultiheadAttention need: [L B C] -> [B L C]
        out = self.hardtanh(out)

        out = self.proj(out).permute(1,0,2) 
        out = out + query
        out = self.norm(out)

        return out




class ParallelTransformer(PipelineParallelBaseModel):
    """Transformer class."""

    def __init__(
        self,
        cfg,
        dtype: torch.dtype=torch.float16,
    ) -> None:
        super(ParallelTransformer, self).__init__()

        self.cfg = cfg
        if gpc.config.model_parallel.use_cpu_initialization:
            self.device = torch.device('cpu')
        else:
            self.device = torch.cuda.current_device()
        self.dtype = dtype
        self.fp32_residual_connection = gpc.config.common.fp32_residual_connection

        ##--------------------------- add cross attention -------------------
        # self.cross_indexs = [1,3,5]

        self.cross_indexs = None
        if cfg.num_layers==2:
            self.cross_indexs = [1]
        elif cfg.num_layers==6:
            self.cross_indexs = [1,3,5]
        else:
            self.cross_indexs = [1,6,12,18]
            
        self.cross_attns = nn.ModuleList()
        for i in self.cross_indexs:
            # self.cross_attns.append(checkpoint_wrapper(CrossAtten_FFN(2048, 32))) 
            self.cross_attns.append(CrossAtten_FFN(2048, 32))

        ##--------------------------- add cross attention -------------------
        self.num_layers, self.layers = self.build_layers(cfg, cfg.num_layers)

        self.hardtanh = nn.Hardtanh(min_val=-32.0, max_val=32.0)

        if self.post_process:
            self.layernorm = LayerNorm(
                cfg.embed_dim,
                eps=cfg.layernorm_epsilon,
                dtype=dtype,
                device=self.device,
            )

    def build_layer(
        self,
        *args,
        **kwargs
    ):
        global_index = int(kwargs.get("global_index"))
        
        layer = ModelParallelTransformerEncoderLayer(
                    layer_number=global_index,
                    self_attention=True,
                    encoder_decoder_attention=False,
                    attention_mask_func=attention_mask_func,
                    cfg=self.cfg
                    # skip_last_bias_add=skip_last_bias_add
                )
        return layer
    def post_process(self, x):
        hidden_states = self.layernorm(hidden_states)
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(self.dtype)
        # reverting data format change [s b h] --> [b s h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params = None,
        kv_hidden_states = None,
        
        row_col_positions = None,
        row_same_mask = None,
        col_same_mask = None,
    ):
        kv_hidden_states_16 = kv_hidden_states
        hs_list = list()
        if not self.pre_process:
            hidden_states = self.input_tensor
        for index in range(self.num_layers):
            global_index = getattr(self.layers[index], 'global_index', -1)
            assert global_index != -1, f"The global_index attribute should be set for layers[{index}]"

            ##--------------------------- add cross attention -------------------
            
            if index in self.cross_indexs:
                idx = self.cross_indexs.index(index)   #L,B，C                L,B，C
                hidden_states = self.cross_attns[idx](query=hidden_states, kv=kv_hidden_states_16)

            hidden_states = self.layers[index](
                hidden_states=hidden_states,
                padding_mask=None,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
                inference_params=inference_params,
                row_col_positions = row_col_positions,
                row_same_mask = row_same_mask,
                col_same_mask = col_same_mask,
            )
            hidden_states = self.hardtanh(hidden_states)
            
            # 添加一个2D 逻辑位置编码

            if not self.fp32_residual_connection:
                hs_list.append(self.layernorm(hidden_states).transpose(0, 1).contiguous())
            else:
                hs_list.append(self.layernorm(hidden_states).to(self.dtype).transpose(0, 1).contiguous())
            

        if self.post_process:
            hidden_states = self.layernorm(hidden_states)
            if self.fp32_residual_connection:
                hidden_states = hidden_states.to(self.dtype)
            # reverting data format change [s b h] --> [b s h]
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        # print(self.pre_process, self.post_process, self.fp32_residual_connection,self.cfg.cfgi_rope)
        return hidden_states, hs_list




class ModelParallelIPTModel(BaseHulkModel,PipelineParallelBaseModel):
    def __init__(self, cfg: IPTConfig):
        super(ModelParallelIPTModel, self).__init__()

        self.embed_dim = cfg.embed_dim
        self.num_layers = cfg.num_layers
        self.parallel_output = cfg.parallel_output
        self.init_method = partial(torch.nn.init.normal_, mean=0.0, std=cfg.init_method_std)
        self.dtype = torch.float32
        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16

        # Embeddings
        self.embedding= None
        if self.pre_process or self.post_process:
            # Embeddings
            self.embedding = EmbeddingEx_cfgi(
                self.embed_dim,
                cfg.padded_vocab_size,
                cfg.max_position_embeddings,
                cfg.hidden_dropout_p,
                self.dtype,
                self.init_method,
            )
            # (NOTE by wlhu5) Because using `self.embedding.word_embeddings.weight` out of module in
            # `parallel_lm_logits`, zero hooks does not work properly, so we entrusts the gathering
            # and sharding of `self.embedding.word_embeddings.weight` to current module.

        # Transformer
        self.transformer = ParallelTransformer(
            cfg, 
            dtype=self.dtype,
        )

    @classmethod
    def build_model(cls, cfg: IPTConfig, task=None):
        """Build a new model instance."""
        model = cls(cfg)

        # pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        # if pipeline_size > 1:
        #     wrapper = PipelineSharedModuleWrapper([0, pipeline_size-1])
        #     if gpc.is_pipeline_first_stage():
        #         wrapper.register_module(model.embedding)

        #     elif gpc.is_pipeline_last_stage():
        #         wrapper.register_module(model.embedding)

        return model
    
    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        self.transformer.set_input_tensor(input_tensor[0])

    def forward(
        self, 
        tokens, 
        position_ids, 
        attention_mask,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params = None,
        return_hidden_states = False,
        row_positions = None
    ):
        if self.pre_process:
            hidden_states = self.embedding(
                tokens, 
                row_positions,
            )
        else:
            hidden_states = None

        # Transformer.
        hidden_states, hs_list = self.transformer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            seq_lengths=seq_lengths,
            inference_params=inference_params,
            row_positions=row_positions,
        )

        if self.post_process:
            logits_parallel = parallel_lm_logits(
                hidden_states,
                self.embedding.word_embeddings.weight,
                self.parallel_output,
            )

            if return_hidden_states:
                return logits_parallel, hidden_states
            else:
                return logits_parallel
        
        return hidden_states, hs_list
