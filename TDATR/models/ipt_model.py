import os
from typing import Optional
from functools import partial
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from TDATR.models.modules.linear_layer import VocabParallelEmbedding

from TDATR.models.modules.transformer_layer_effiency import ModelParallelTransformerEncoderLayer, PipelineParallelBaseModel, TransformerEncoderLayerConfig


from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
from TDATR.models.modules.base_model import BaseHulkModel
from TDATR.models.modules.transformer_layer_effiency import attention_mask_func
from TDATR_utils.models import parallel_lm_logits

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
            self.layernorm = nn.LayerNorm(
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
    ):
        kv_hidden_states_16 = kv_hidden_states
        hs_list = list()
        if not self.pre_process:
            hidden_states = self.input_tensor
        for index in range(self.num_layers):
            global_index = getattr(self.layers[index], 'global_index', -1)
            assert global_index != -1, f"The global_index attribute should be set for layers[{index}]"
            # if global_index == self.cfg.num_layers:
            #     continue

            ##--------------------------- add cross attention -------------------
            
            if index in self.cross_indexs:
                idx = self.cross_indexs.index(index)   #L,B，C                L,B，C

                hidden_states = self.cross_attns[idx](query=hidden_states, kv=kv_hidden_states_16)
                # if index in [1, 5]:
                #     idx = self.cross_indexs.index(index)   #L,B，C                L,B，C
                #     hidden_states = self.cross_attns[idx](query=hidden_states, kv=kv_hidden_states_32)
                # else:
                #     idx = self.cross_indexs.index(index)   #L,B，C                L,B，C
                #     hidden_states = self.cross_attns[idx](query=hidden_states, kv=kv_hidden_states_16)
            
            hidden_states = self.layers[index](
                hidden_states=hidden_states,
                padding_mask=None,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
                inference_params=inference_params
            )
            hidden_states = self.hardtanh(hidden_states)
            
            if not self.fp32_residual_connection:
                hs_list.append(self.layernorm(hidden_states).transpose(0, 1).contiguous())
            else:
                hs_list.append(self.layernorm(hidden_states).to(self.dtype).transpose(0, 1).contiguous())

        if self.post_process:
            hidden_states = self.layernorm(hidden_states)
            if self.fp32_residual_connection:
                hidden_states = hidden_states.to(self.dtype)
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        
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
            self.embedding = Embedding(
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

        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
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
    ):
        if self.pre_process:
            hidden_states = self.embedding(
                tokens, 
                position_ids,
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
