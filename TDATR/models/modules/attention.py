import os
from typing import Optional
from functools import partial
from dataclasses import dataclass, field
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from TDATR.models.modules.linear_layer import VocabParallelEmbedding
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
 


from typing import Optional, Callable, Tuple

import logging
import warnings

logger= logging.getLogger(__name__)



class EmbeddingEx(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        dtype,
        init_method,
        with_position=True
    ):
        super(EmbeddingEx, self).__init__()

        self.embed_dim = embed_dim
        self.init_method = init_method

        self.sequence_parallel = gpc.config.model_parallel.sequence_parallel
        self.use_cpu_initialization = gpc.config.model_parallel.use_cpu_initialization
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
        self.with_position = with_position
        self.position_embeddings = None
        if self.with_position:
            # Position embedding (serial).
            self.position_embeddings = torch.nn.Embedding(
                num_embeddings = max_sequence_length,
                embedding_dim = self.embed_dim,
                dtype=dtype,
                device=device
            )
            self.init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def forward_(self, tokens, position_ids, save_mem=False):
        

        words_embeddings = self.word_embeddings(tokens)
        if self.with_position:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings= words_embeddings

        embeddings = embeddings.transpose(0, 1).contiguous()

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def forward(self, tokens, position_ids):
        return self.forward_(tokens, position_ids, save_mem=False)

class RotaryPositionalTransform(torch.nn.Module):
    def __init__(self, dim, base=10000, dtype= torch.float16):
        super().__init__()
        self.dtype = dtype
        self.base= base
        self.dim=dim
        # inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # inv_freq=inv_freq
        # self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        
    def get_cos_sin(self, x):
        """
        Args:
            x: Input x with B XT X  C
            seq_len: Sequence length of input x
        """
        seq_len= x.shape[1]
        if seq_len != self.seq_len_cached:
            # these initialization must be in float32
            self.seq_len_cached = seq_len
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            inv_freq= inv_freq.to(x.device)
            t = torch.arange(seq_len, device=x.device).float()
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, :,  :].to(self.dtype)
            self.sin_cached = emb.sin()[None,  :, :].to(self.dtype)
        return self.cos_cached, self.sin_cached
    
    def forward(self, q,k):
        """
            q: B X T_q  X C
            k: B X T X  C
            while training, T_q=T, while inference, T_q=1 
        """
        
        cos, sin = self.get_cos_sin(k)
        if q.shape[1] > k.shape[1]:
            raise ValueError(f"q shape {q.shape[1]} bigger than k {k.shape[1]}")
        if q.shape[1] < k.shape[1]:
            cos_q,sin_q= cos[:, -q.shape[1]:], sin[:, -q.shape[1]:]
        else:
            cos_q, sin_q= cos, sin
        q= q*cos_q +self.rotate_half(q)*sin_q
        k= k*cos + self.rotate_half(k)*sin
        return q,k

    # rotary pos emb helpers:
    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in earlier torch versions




class CoreAttentionExpand(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        layer_scaling: Optional[int]=-1,
        self_attention: Optional[bool]=False,
        encoder_decoder_attention: Optional[bool]=False,
        use_rope=False,
        linear_attention= False,
        sparse_local_size= 1024,
        sparse_stride_size= 128,
        sparse_local_extra= 128
        # attention_mask_func: Optional[Callable]=attention_mask_func,
    ) -> None:
        super(CoreAttentionExpand, self).__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.layer_scaling = max(1, layer_scaling)
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        # self.attention_mask_func = attention_mask_func

        cfg = gpc.config.model
        self.apply_query_key_layer_scaling = cfg.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = cfg.attention_softmax_in_fp32
        self.fp16 = gpc.config.common.fp16
        self.sequence_parallel = gpc.config.model_parallel.sequence_parallel

        # world_size = gpc.get_world_size(ParallelMode.TENSOR)
        
        

        self.use_rope= use_rope
        self.linear_attention= linear_attention

        self.rotary_embedding= None
        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        if self.use_rope:
            self.rotary_embedding= RotaryPositionalTransform(self.head_dim,dtype=self.dtype)
        
        self.norm_factor = 1./ math.sqrt(head_dim)

        if linear_attention:
            raise ValueError("linear attention is used, which is not supported for autoregressvie model now")
        self.sparse_local_size= sparse_local_size
        self.sparse_stride_size= sparse_stride_size
        self.sparse_local_extra= sparse_local_extra
        self.use_sparse= False
        if self.sparse_local_size>0:
            self.use_sparse = True
            assert self.sparse_stride_size >0 and self.sparse_stride_size <=self.sparse_local_size
        self.zero_k, self.zero_v= None,None
        if self.use_sparse:
            world_size = gpc.get_world_size(ParallelMode.TENSOR)
            self.zero_k = nn.Parameter(torch.empty(world_size, self.num_head,self.head_dim))
            self.zero_v= nn.Parameter(torch.empty(world_size, self.num_head,self.head_dim))
            init.normal_(self.zero_k, std=0.01)
            init.normal_(self.zero_v, std=0.01)
        self.attention_dropout = torch.nn.Dropout(cfg.attention_dropout_p, inplace=False)
    
    def dense_attention(self, q,k,v,attention_mask):
        # B_H, T, D
        tgt_len,src_len= q.shape[1], k.shape[1]
        bsz= q.shape[0]// self.num_head
        attn_weights= torch.bmm(q, k.transpose(1,2))
        if attention_mask is not None:
            attn_weights=attn_weights.masked_fill_(attention_mask, -10000.0)
        
        attention_probs= F.softmax(attn_weights, dim=-1)
        
        attention_probs = self.attention_dropout(attention_probs)
        # Context layer
        assert v is not None
        context = torch.bmm(attention_probs, v)
        assert list(context.size()) == [
            bsz * self.num_head,
            tgt_len,
            self.head_dim,
        ]
        output = context.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return output
    
    def gen_global_mask(self, k):
        with torch.no_grad():
            qlen= k.shape[1]
            klen= qlen//self.sparse_stride_size+1
            qrange= torch.arange(qlen)
            krange= torch.arange(klen)*self.sparse_stride_size
            mask = qrange.unsqueeze(1) < krange.unsqueeze(0)
            mask= mask.unsqueeze(0).to(k.device)
        return mask
    
    def gen_global_kvmask(self, q,k,v, attention_mask):
        B,T,D= k.shape
        kg= k.reshape(B, T//self.sparse_stride_size, self.sparse_stride_size, D).mean(dim=2)
        vg= v.reshape(B, T//self.sparse_stride_size, self.sparse_stride_size, D).mean(dim=2)
        # if gpc.get_global_rank()==0:
        #     
        
        curr_rank= gpc.get_local_rank(ParallelMode.TENSOR)
        bsz= B// self.num_head
        # curr_rank=0
        kzero,vzero= self.zero_k[curr_rank], self.zero_v[curr_rank]
        kzero= kzero.repeat(bsz,1).view(B,1,D)
        vzero= vzero.repeat(bsz,1).view(B,1,D)
        kg= torch.cat([kzero,kg], dim=1)
        vg= torch.cat([vzero,vg],dim=1)
        global_mask = None
        if attention_mask is not None:
            global_mask = self.gen_global_mask(q)
        return kg,vg, global_mask


    
    def sparse_attention(self, q,k,v,attention_mask):
        a= torch.randn(4,4)
        a.numel
        def attention(q,k,v, mask,ignore_len=0):
            attn_weights= torch.bmm(q,k.transpose(1,2))
            if mask is not None:
                attn_weights = attn_weights.masked_fill_(mask, -10000.0)
            attn_weights = F.softmax(attn_weights, dim=-1)
            if ignore_len>0:
                filler= torch.arange(attn_weights.shape[1],device= q.device)<ignore_len
                filler= filler[None, :, None]
                attn_weights= attn_weights.masked_fill(filler, 0.0)
            attn_weights= self.attention_dropout(attn_weights)
            attn=  torch.bmm(attn_weights,v)
            return attn
        
        def reshape_kv(k):
            B,T,D= k.shape
            snum = T//self.sparse_local_size
            if self.sparse_local_extra==0:
                kl=k.reshape(-1,self.sparse_local_size, D)
                return kl
            kl = k.reshape(B,snum, self.sparse_local_size, D)
            khis = kl[:,:, -self.sparse_local_extra:]
            khis0 = torch.zeros(B,1, self.sparse_local_extra,D).to(khis)
            khis= torch.cat([khis0,khis[:,:-1]],dim=1)
            kl = torch.cat([khis,kl], dim = 2)
            kl = kl.reshape(B*snum, self.sparse_local_extra+ self.sparse_local_size, D)
            return kl
            # k = k.transpose(0,1)
            # k= F.pad(k, (0,0,0,0,128,0), value=0.0)
            # snum = T//self.sparse_local_size
            # # 4,B,D,T
            # k = k.as_strided(
            #     (snum, B,D,self.sparse_local_size+self.sparse_local_extra),
            #     (B*D*self.sparse_local_size,D,1,B*D)
            # )
            # k= k.permute(1,0,3,2).contiguous().view(B*snum, -1, D)
            # return k
        def get_local_mask(global_mask, k):
            local_mask= attention_mask[:, -self.sparse_local_size:, -self.sparse_local_size:]
            if self.sparse_local_extra >0:
                B,T,D= k.shape
                snum = T//self.sparse_local_size
                his_mask = torch.zeros(B,snum, self.sparse_local_size, self.sparse_local_extra).to(global_mask)
                # mask history from first block
                his_mask[:,0]=True
                his_mask= his_mask.view(-1, self.sparse_local_size, self.sparse_local_extra)
                local_mask= local_mask.repeat(B*snum, 1,1)
                local_mask= torch.cat([his_mask, local_mask],dim=-1)
            return local_mask

        # B_H, T, D
        tgt_len,src_len= q.shape[1], k.shape[1]
        bsz= q.shape[0]// self.num_head
        # calculate local attention
        if tgt_len ==1:
            local_len = src_len %self.sparse_local_size + self.sparse_local_extra +1
            k_local,v_local = k[:,-local_len], v[:,-local_len]
            # B_H,T,D
            local_attn = attention(q, k_local, v_local, None)
        else:
            assert tgt_len== src_len, f"tgt{tgt_len} not equal to src size {src_len} for teacher forcing"
            assert tgt_len % self.sparse_local_size ==0, f"seq length {tgt_len} not split by local_size {self.sparse_local_size}, maybe inference with wrong prompt processing"
            B,T,D= q.shape
            pnum = tgt_len // self.sparse_local_size
            ql=q.reshape(-1, self.sparse_local_size, D)
            kl = reshape_kv(k)
            vl= reshape_kv(v)
            local_mask= None
            if attention_mask is not None:
                local_mask= get_local_mask(attention_mask,k)
            local_attn= attention(ql,kl,vl, local_mask)
            local_attn= local_attn.reshape(B,T,D)
        # calculate long history stride attention
        kg,vg,global_mask= self.gen_global_kvmask(q,k,v, attention_mask)
        
        global_attn= attention(q,kg,vg, global_mask)

        output= local_attn + global_attn
        output= output.transpose(0,1).contiguous().view(tgt_len, bsz, -1)
        return output

    def forward(
        self,
        q, k, v,
        padding_mask,
        attention_mask,
    ):
        bsz, tgt_len = q.size(1), q.size(0)
        src_len = k.size(0)
        # [tgt_len, batch, num_head * head_dim] -> [batch * num_head, tgt_len, head_dim]
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_head, self.head_dim) # [tgt_len, batch * num_head, head_dim]
            .transpose(0, 1)  # [batch * num_head, tgt_len, head_dim]
        )
        # [src_len, batch, num_head * head_dim] -> [batch * num_head, src_len, head_dim]
        k = (
            k.contiguous()
            .view(-1, bsz * self.num_head, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(-1, bsz * self.num_head, self.head_dim)
            .transpose(0, 1)
        )
        
        if self.use_rope:
            q,k= self.rotary_embedding(q,k)
        
        q= q*self.norm_factor
        
        if not self.use_sparse:
            output= self.dense_attention(q,k,v, attention_mask)
        else:
            output= self.sparse_attention(q,k,v, attention_mask)
        return output

    def linear_attention(self, q, k, v):
        # only works with triangle mask internal, \phi(q) \dot cumsum(\phi(k) \dot v)
        # input: B_H, T,D
        tgt_len,src_len= q.shape[1], k.shape[1]
        bsz= q.shape[0]// self.num_head
        q= F.elu(q)+1
        k= F.elu(k)+1
        #B,T,D,D
        kv= torch.einsum('btd,btm->btdm',k,v)
        kv_cum = torch.cumsum(kv, dim=1)
        k_cum= torch.cumsum(k, dim=1)
        if tgt_len != src_len:
            kv_cum= kv_cum[:,-tgt_len:]
            k_cum= k_cum[:,-tgt_len:]
        den = torch.einsum('btd,btd->bt', q, k_cum)
        num= torch.einsum('btd,btdm->btm', q, kv_cum)
        output= num / den.unsqueeze(-1)
        output = output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return output
