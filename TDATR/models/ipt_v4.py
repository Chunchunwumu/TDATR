import logging
import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from TDATR.models.modules.transformer_layer_effiency import ModelParallelTransformerEncoderLayer
from TDATR_utils.global_context import global_context as gpc

from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.global_variables import ChoiceEnum

ModelParallelTransformerEncoderLayer

from TDATR.modules.attention import  EmbeddingEx

from TDATR.modules.block_attn_v2 import SparseSelfAttention
from TDATR.modules.layer import ModelParallelMLP

from .ipt_model import (
    IPTConfig,
    ModelParallelIPTModel,
    ParallelTransformer,
    parallel_lm_logits,
)

logger = logging.getLogger(__name__)
ATTENTION_TYPE_CHOICES = ChoiceEnum(["softmax", "linear"])
POSITION_EMBEDDING_CHOICES = ChoiceEnum(["none", "rope", "xpos"])


@dataclass
class IPTAttConfigV4(IPTConfig):
    attention_type: Optional[ATTENTION_TYPE_CHOICES] = field(
        default="softmax",
        metadata={"help": "attention types: current support: softmax, linear "},
    )
    position_type: Optional[POSITION_EMBEDDING_CHOICES] = field(
        default="rope",
        metadata={
            "help": "position embedding types: current support: xpos, none, rope "
        },
    )
    sparse_local_size: Optional[int] = field(
        default=-1,
        metadata={"help": "local size for sparse attention, -1 for full attention"},
    )
    # sparse_stride_size:Optional[int] = field(
    #     default= -1,
    #     metadata={"help": "stride size for sparse attention long history"}
    # )
    # sparse_local_extra:Optional[int] = field(
    #     default= -1,
    #     metadata={"help":"local size for sparse attention, -1 for full attention"}
    # )
    gate_gelu: Optional[bool] = field(
        default=False, metadata={"help": "use gate gelu for mlp"}
    )
    use_flash: Optional[bool] = field(
        default=True, metadata={"help": "use flash attention or not"}
    )
    use_flash_v2: Optional[bool] = field(
        default=False, metadata={"help": "use flash attention v2 or not"}
    )
    use_sparse: Optional[bool] = field(
        default=False, metadata={"help": "use flash sparse attention or not"}
    )
    use_smooth: Optional[bool] = field(
        default=False, metadata={"help": "use smooth sparse attention or not"}
    )
    use_naiive: Optional[bool] = field(
        default=False,
        metadata={
            "help": "use naiive attention or flash attention when use sparse attention"
        },
    )
    use_offical_fa2: bool = field(
        default=False,
        metadata={
            "help": "use rope and flash attention-2 implemented by `flash-attn==2.3.3`"
        }
    )
    rope_base: int = field(
        default=10000,
        metadata={
            'help': "base value for rope"
        }
    )


class IPTV4Model(ModelParallelIPTModel):
    def __init__(self, cfg: IPTConfig):
        super(IPTV4Model, self).__init__(cfg)
        # self.sparse_stride_size= cfg.sparse_stride_size
        logger.info(f"=> Training IPT-V4 with {self.dtype}")
        self.sparse_local_size = cfg.sparse_local_size
        self.cfg = cfg
        if self.cfg.position_type != "none" and self.embedding is not None:
            logger.info(
                f"position_type is {self.cfg.position_type}, use EmbeddingEx without position embedding"
            )
            self.embedding = EmbeddingEx(
                self.embed_dim,
                cfg.padded_vocab_size,
                cfg.max_position_embeddings,
                cfg.hidden_dropout_p,
                self.dtype,
                self.init_method,
                with_position=False,
            )
        skip_last_bias_add = True
        attention_dropout_p = gpc.config.model.attention_dropout_p
        self.recompute_granularity = gpc.config.model_parallel.recompute_granularity

        

        for nl, layer in enumerate(self.transformer.layers):
            if isinstance(layer, ModelParallelTransformerEncoderLayer):
                att = layer.attention
                if nl == 0:
                    logger.info(
                        "use proto_flash.SparseSelfAttention for CoreAttention"
                    )
                core_attention = SparseSelfAttention(
                    head_dim=att.head_dim,
                    num_head=att.num_heads_per_partition,
                    num_kv_head=att.num_kv_heads_per_partition,
                    max_seq_length=gpc.config.task.seq_length,
                    local_size=gpc.config.model.sparse_local_size,
                    position_type=self.cfg.position_type,
                    rope_base=self.cfg.rope_base,
                    attention_dropout_p=attention_dropout_p,
                    dtype=self.dtype,
                    use_naiive=gpc.config.model.use_naiive,
                    use_smooth=gpc.config.model.use_smooth,
                    use_fa_v2=gpc.config.model.use_flash_v2,
                )
                    
                layer.attention.core_attention = core_attention
                

                layer.mlp = ModelParallelMLP(
                    embed_dim=layer.embed_dim,
                    mlp_embed_dim=layer.mlp_embed_dim,
                    dtype=layer.dtype,
                    skip_last_bias_add=skip_last_bias_add,
                    gate_gelu=cfg.gate_gelu,
                )

    def reset_attention_mask(self, tokens):
        seq_len = tokens.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((seq_len, seq_len), device=tokens.device)
        ).bool()
        return attention_mask.unsqueeze(0)

    def forward(
        self,
        tokens,
        position_ids,
        attention_mask,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params=None,
        return_hidden_states=False,
        kv_hidden_states=None,
        task=None
    ):

        if self.pre_process:
            hidden_states = tokens.permute(1, 0, 2)
        else:
            hidden_states = None
        hidden_states,hs_list = self.transformer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inference_params=inference_params,
            kv_hidden_states = kv_hidden_states
        )

        if self.post_process:
            logits_parallel = parallel_lm_logits(
                hidden_states,
                self.embedding.word_embeddings.weight,
                self.parallel_output,
            )

            if return_hidden_states:
                return logits_parallel, (hidden_states,hs_list)
            else:
                return logits_parallel
        
        return hidden_states


        

