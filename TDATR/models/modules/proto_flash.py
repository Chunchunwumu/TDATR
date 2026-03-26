"""
implementation of flash attention with local sparse as prototype, which just follow the algorithm by einsum etc.
Added by danliu, temporaryly
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import einops
from torch.autograd.function import Function
from .attention import RotaryPositionalTransform
from .xpos import XPOS
import warnings

EPSILON = 1e-6


class CausalLocalAttentionFunction(Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, local_size, q_bucket_size, k_bucket_size,dropout_p):
        """ Algorithm 2 in the paper """
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        device = q.device
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)


        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            all_row_sums.split(q_bucket_size, dim = -2),
            all_row_maxes.split(q_bucket_size, dim = -2),
        )

        for ind, (qc, oc, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size + qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size
                if (k_start_index  < q_start_index -local_size) or (k_start_index >q_start_index+ q_bucket_size):
                    continue
                attn_weights = torch.einsum('... i d, ... j d -> ... i j', qc, kc) 
                q_start_index< k_start_index+k_bucket_size-1
                if k_start_index +k_bucket_size-1 >q_start_index:
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)


                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                # apply dropout
                if dropout_p>0:
                    exp_weights = F.dropout(exp_weights, p= dropout_p)

                exp_values = torch.einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (local_size, q_bucket_size, k_bucket_size, dropout_p)
        ctx.save_for_backward(q, k, v, o, lse, rng_state)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        local_size, q_bucket_size, k_bucket_size,dropout_p = ctx.args
        q, k, v, o, lse, rng_state = ctx.saved_tensors
        cur_rng_state= None
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            lse.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        for ind, (qc, oc, doc, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                dk.split(k_bucket_size, dim = -2),
                dv.split(k_bucket_size, dim = -2),
            )
           
            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size
                if (k_start_index  < q_start_index -local_size) or (k_start_index >q_start_index+ q_bucket_size):
                    continue

                attn_weights = torch.einsum('... i d, ... j d -> ... i j', qc, kc)

                if  q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                p = torch.exp(attn_weights.float() - lsec).to(qc.dtype)
                if dropout_p >0:
                    p= F.dropout(p, dropout_p)

                dv_chunk = torch.einsum('... i j, ... i d -> ... j d', p, doc)
                dp = torch.einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim = -1, keepdims = True)
                ds = p * (dp - D)

                dq_chunk = torch.einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = torch.einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)

        return dq, dk, dv, None, None, None, None

def get_blockmask(seqlen, local_size=256, blocksize_q=16,blocksize_k= 128):
    assert seqlen % blocksize_k==0, f"seqlen {seqlen} not dividable by blocksize_k{blocksize_k}"
    assert blocksize_k %blocksize_q==0, f"k_block{blocksize_k} should be times of blocksize_q{blocksize_q}"
    qlen,klen= seqlen//blocksize_q, seqlen//blocksize_k
    qrange= torch.arange(qlen)
    krange= torch.arange(klen)
    # (k_start_index  < q_start_index -local_size) or (k_start_index >q_start_index+ q_bucket_size)
    import pdb
    # mask1=  krange[None,:]*blocksize_k>= qrange[:,None]*blocksize_q+blocksize_q
    # mask2= krange[None,:]*blocksize_k <= qrange[:, None]*blocksize
    mask1=   qrange[:,None]*blocksize_q <krange[None,:]*blocksize_k
    mask2= qrange[:, None]*blocksize_q  >=(krange[None,:])*blocksize_k +local_size
    mask= mask1|mask2
    return mask


class SparseSelfAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        max_seq_length=4096,
        local_size= 1024,
        block_q=16,
        block_k=128,
        position_type="rope",
        attention_dropout_p=0.1,
        dtype= torch.float16,
        use_naiive=False,
        **kwargs
    ) -> None:
        super(SparseSelfAttention, self).__init__()
        assert position_type in ["rope","xpos", "none"], f"now only support rope, xpos, none"
        self.head_dim = head_dim
        self.num_head = num_head
        self.use_rope= position_type != "none"
        
        self.rotary_embedding= None
        if position_type== "rope":
            self.rotary_embedding= RotaryPositionalTransform(self.head_dim, dtype=dtype)
        elif position_type =="xpos":
            self.rotary_embedding = XPOS(self.head_dim, dtype=dtype)
        
        self.use_naiive=use_naiive
        # if torch.cuda.get_device_capability(0)[0] < 7.5:
        #     warnings.warn("NOTE: your device does NOT support flash attention, back to naiive")
        #     self.use_naiive=True
        
        self.norm_factor = 1./ math.sqrt(head_dim)
        # cfg = gpc.config.model
        # self.dropout_p= cfg.attention_dropout_p
        self.dropout_p= attention_dropout_p

        self.block_q, self.block_k=block_q, block_k
        self.local_size= local_size
        layout= get_blockmask(max_seq_length,local_size, self.block_q, self.block_k)
        self.register_buffer("layout", layout)
        # transform to fairseq mask: 0 for normal and 1 for ignore
        fullmask= self.expand_blockmask(layout)
        self.register_buffer('fullmask', fullmask)

    
    def expand_blockmask(self, layout):
        qlen,klen= layout.shape
        seqlen= qlen* self.block_q
        assert seqlen == self.block_k*klen
        fullmask= layout.view(qlen, 1, klen, 1).repeat(1, self.block_q, 1, self.block_k)
        fullmask= fullmask.reshape(seqlen, seqlen)
        #  0 for good,1 for padding
        seqid= torch.arange(seqlen, dtype= torch.long)
        causal_mask= seqid[:,None]<seqid[None,:]
        removed= (~fullmask)& causal_mask
        import pdb 
        fullmask= fullmask| causal_mask
        return fullmask
    
    def forward(
        self,
        q, k, v,
        padding_mask,
        attention_mask,
        **kwargs
    ):
        src_len, tgt_len = k.shape[0], q.shape[0]
        bsz= q.shape[1]
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
        q = q*self.norm_factor
        if self.use_rope:
            q,k= self.rotary_embedding(q,k)
        
        # self.dropout_p=0.0
        # out1= self.fwd_naiive(q,k,v, attention_mask)
        # out2= self.fwd_flash(q,k,v)
        # diff= out1-out2
        if self.use_naiive:
            output= self.fwd_naiive(q,k,v)
        else:
            output= self.fwd_flash(q,k,v)
        return output
    
    def fwd_flash(self,q,k,v):
        tgt_len,src_len= q.shape[1], k.shape[1]
        bsz= q.shape[0]// self.num_head
        out = CausalLocalAttentionFunction.apply(q, k, v, self.local_size, self.block_q, self.block_k,self.dropout_p)
        out= out.transpose(0,1).contiguous().view(tgt_len, bsz,-1)
        return out
    
    def fwd_naiive(self, q,k,v):
        tgt_len,src_len= q.shape[1], k.shape[1]
        bsz= q.shape[0]// self.num_head
        attn_weights= torch.bmm(q, k.transpose(1,2))
        fullmask= self.fullmask[-tgt_len:, -src_len:]

        attn_weights=attn_weights.masked_fill_(fullmask, -10000.0)
        
        attention_probs= F.softmax(attn_weights, dim=-1)
        
        attention_probs = F.dropout(attention_probs, p= self.dropout_p, training= self.training) 
        context = torch.bmm(attention_probs, v)
        
        output = context.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return output