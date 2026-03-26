import logging
import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ChoiceEnum

from TDATR.modules.attention import EmbeddingEx_cfgi
from TDATR.modules.dense_attn import FlashCoreAttention,FlashCoreAttention_row_col
from TDATR.modules.layer import ModelParallelMLP

from .ipt_model_cfgi import (
    IPTConfig,
    ModelParallelIPTModel,
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
        logger.info(f"=> Training IPT-V4 with {self.dtype}")
        self.sparse_local_size = cfg.sparse_local_size
        self.cfg = cfg
        if self.cfg.position_type != "none" and self.embedding is not None:
            logger.info(
                f"position_type is {self.cfg.position_type}, use EmbeddingEx without position embedding"
            )
            self.embedding = EmbeddingEx_cfgi(
                self.embed_dim,
                cfg.padded_vocab_size,
                cfg.max_position_embeddings,
                cfg.hidden_dropout_p,
                self.dtype,
                self.init_method,
                with_position=False,
            )
        skip_last_bias_add = True
        if gpc.config.distributed_training.ddp_backend == "fully_sharded":
            skip_last_bias_add = False
        
        attention_dropout_p = gpc.config.model.attention_dropout_p
        
        for nl, layer in enumerate(self.transformer.layers):
            att = layer.attention
            core_attention = FlashCoreAttention(
                    head_dim=att.head_dim,
                    num_head=att.num_heads_per_partition,
                    max_seq_length=gpc.config.task.seq_length,
                    position_type="none",
                    rope_base=self.cfg.rope_base,
                    dtype=self.dtype,
                    attention_dropout_p=attention_dropout_p,
                )
            layer.attention.core_attention = core_attention
            layer.row_attention.core_attention = FlashCoreAttention_row_col(
                    head_dim=att.head_dim,
                    num_head=att.num_heads_per_partition,
                    max_seq_length=gpc.config.task.seq_length,
                    position_type="none",
                    rope_base=self.cfg.rope_base,
                    dtype=self.dtype,
                    attention_dropout_p=attention_dropout_p,
                )
            layer.col_attention.core_attention = FlashCoreAttention_row_col(
                    head_dim=att.head_dim,
                    num_head=att.num_heads_per_partition,
                    max_seq_length=gpc.config.task.seq_length,
                    position_type="none",
                    rope_base=self.cfg.rope_base,
                    dtype=self.dtype,
                    attention_dropout_p=attention_dropout_p,
                )
            

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
        row_positions = None
    ):
        attention_mask = None
        return super().forward(
            tokens, position_ids, attention_mask, seq_lengths, inference_params, return_hidden_states,row_positions=row_positions
        )
