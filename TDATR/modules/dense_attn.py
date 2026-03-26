from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
# import flash_attn
from einops import rearrange

from .xpos import XPOS,RotaryPositionalTransform
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
import logging 
import warnings
logger= logging.getLogger(__name__)

class FlashCoreAttention(nn.Module):
    full_mask_cached: Dict[Tuple[int, torch.device], torch.Tensor] = dict()
    causal_mask_cached: Dict[Tuple[int, torch.device], torch.Tensor] = dict()
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        max_seq_length:int=4096,
        num_kv_head: Optional[int]=None,
        position_type="none",
        dtype= torch.float16,
        attention_dropout_p=0.1,
        use_fa_v2=False,
        name=None,
        **kwargs
    ) -> None:
        super(FlashCoreAttention, self).__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.num_kv_head = num_kv_head if num_kv_head is not None else num_head
        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.max_seq_length=max_seq_length
        self.use_rope= position_type != "none"
        self.use_fa_v2 = use_fa_v2

        self.rotary_embedding= None
        if position_type== "rope":
            self.rotary_embedding= RotaryPositionalTransform(self.head_dim, dtype=dtype)
        elif position_type =="xpos":
            self.rotary_embedding = XPOS(self.head_dim,scale_base=max_seq_length, dtype=dtype)
            
        self.use_naiive = gpc.config.model.use_naiive
        if gpc.config.common.npu:
            self.name = name
            # if name!="med": 
            self.use_naiive = True
            # logger.info('use Naiive CoreAttention for NPU')
        # maj,minor=torch.cuda.get_device_capability(0)
        # if not (maj==8 and minor==0):
        #     warnings.warn("NOTE: your device does NOT support flash attention, back to naiive")
        #     self.use_naiive=True
        
        self.norm_factor = 1./ math.sqrt(head_dim)
        # cfg = gpc.config.model
        # cfg.attention_dropout_p
        self.dropout_p= attention_dropout_p

    def get_fullmask(self, device: torch.device, dtype=torch.bool) -> torch.BoolTensor:
        cache_key = (
            self.max_seq_length,
            device,
        )
        if cache_key not in FlashCoreAttention.full_mask_cached:
            mask = torch.ones(self.max_seq_length, self.max_seq_length, device=device).bool()
            FlashCoreAttention.full_mask_cached[cache_key] = mask
        return FlashCoreAttention.full_mask_cached[cache_key]  

    def get_causal_mask(self, device: torch.device, dtype=torch.bool) -> torch.BoolTensor:
        cache_key = (
            self.max_seq_length,
            device,
        )
        if cache_key not in FlashCoreAttention.causal_mask_cached:
            mask = torch.triu(torch.ones(self.max_seq_length, self.max_seq_length), diagonal=1).bool().to(device)
            FlashCoreAttention.causal_mask_cached[cache_key] = mask
        return FlashCoreAttention.causal_mask_cached[cache_key]  

    def forward(
        self,
        q, k, v,
        padding_mask,
        attention_mask,
        seq_lengths=None,
        inference_params=None,
    ):
        src_len, tgt_len = k.shape[0], q.shape[0]
        bsz = q.shape[1]
        if q.device.type == 'cuda':
            q = (
                q.contiguous()
                .view(
                    tgt_len, bsz, self.num_head, self.head_dim
                )  # [tgt_len, batch * num_head, head_dim]
                .transpose(0, 1)  # [batch * num_head, tgt_len, head_dim]
            )
            # [src_len, batch, num_head * head_dim] -> [batch * num_head, src_len, head_dim]
            k = k.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim).transpose(0, 1)


        elif q.device.type == 'npu':
            q = (
                q.contiguous()
                .view(
                    tgt_len, bsz, self.num_head, self.head_dim
                )  # [tgt_len, batch * num_head, head_dim]
            )
            # [src_len, batch, num_head * head_dim] -> [batch * num_head, src_len, head_dim]
            k = k.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim)
            v = v.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim)
        
        if (self.use_naiive or tgt_len != src_len) and q.device.type == 'npu':
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        use_mask = (attention_mask is not None) and tgt_len == src_len
        output= self.fwd_naiive(q,k,v,attention_mask)
        
        return output

    def fwd_naiive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        atttion_mask=None,
        use_mask: bool = True,
        fullmask: Optional[torch.BoolTensor] = None,
        seq_lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        nhead, hdim = q.shape[-2:]
        bsz = q.shape[0]
        tgt_len, src_len = q.shape[1], k.shape[1]
        # BTHD- >(BH)TD
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, tgt_len, hdim)
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, src_len, hdim)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, src_len, hdim)

        q = q.float() * self.norm_factor
        k = k.float()
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        

        if atttion_mask is not None:
            attn_weights = attn_weights + (atttion_mask.permute(2,0,1)-1)*1e4

        attention_probs = F.softmax(attn_weights, dim=-1).to(v)
        
        context = torch.bmm(attention_probs, v)

        output = context.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return output


class FlashCoreAttention_row_col(nn.Module):
    full_mask_cached: Dict[Tuple[int, torch.device], torch.Tensor] = dict()
    causal_mask_cached: Dict[Tuple[int, torch.device], torch.Tensor] = dict()
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        max_seq_length:int=4096,
        num_kv_head: Optional[int]=None,
        position_type="none",
        dtype= torch.float16,
        attention_dropout_p=0.1,
        use_fa_v2=False,
        name=None,
        **kwargs
    ) -> None:
        super(FlashCoreAttention_row_col, self).__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.num_kv_head = num_kv_head if num_kv_head is not None else num_head
        if gpc.config.common.fp16:
            self.dtype = torch.float16
        elif gpc.config.common.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.max_seq_length=max_seq_length
        self.use_rope= position_type != "none"
        self.use_fa_v2 = use_fa_v2

        self.rotary_embedding= None
        if position_type== "rope":
            self.rotary_embedding= RotaryPositionalTransform(self.head_dim, dtype=dtype)
        elif position_type =="xpos":
            self.rotary_embedding = XPOS(self.head_dim,scale_base=max_seq_length, dtype=dtype)
            
        self.use_naiive = gpc.config.model.use_naiive

        if gpc.config.common.npu:
            self.name = name
            # if name!="med": 
            self.use_naiive = True
            logger.info('use Naiive CoreAttention for NPU')
        # maj,minor=torch.cuda.get_device_capability(0)
        # if not (maj==8 and minor==0):
        #     warnings.warn("NOTE: your device does NOT support flash attention, back to naiive")
        #     self.use_naiive=True
        
        self.norm_factor = 1./ math.sqrt(head_dim)
        # cfg = gpc.config.model
        # cfg.attention_dropout_p
        self.dropout_p= attention_dropout_p

    def get_fullmask(self, device: torch.device, dtype=torch.bool) -> torch.BoolTensor:
        cache_key = (
            self.max_seq_length,
            device,
        )
        if cache_key not in FlashCoreAttention_row_col.full_mask_cached:
            mask = torch.ones(self.max_seq_length, self.max_seq_length, device=device).bool()
            FlashCoreAttention_row_col.full_mask_cached[cache_key] = mask
        return FlashCoreAttention_row_col.full_mask_cached[cache_key]  

    def get_causal_mask(self, device: torch.device, dtype=torch.bool) -> torch.BoolTensor:
        cache_key = (
            self.max_seq_length,
            device,
        )
        if cache_key not in FlashCoreAttention_row_col.causal_mask_cached:
            mask = torch.triu(torch.ones(self.max_seq_length, self.max_seq_length), diagonal=1).bool().to(device)
            FlashCoreAttention_row_col.causal_mask_cached[cache_key] = mask
        return FlashCoreAttention_row_col.causal_mask_cached[cache_key]  

    def forward(
        self,
        q, k, v,
        padding_mask,
        attention_mask,
        seq_lengths=None,
        inference_params=None,
    ):
        src_len, tgt_len = k.shape[0], q.shape[0]
        bsz = q.shape[1]
        if q.device.type == 'cuda':
            q = (
                q.contiguous()
                .view(
                    tgt_len, bsz, self.num_head, self.head_dim
                )  # [tgt_len, batch * num_head, head_dim]
                .transpose(0, 1)  # [batch * num_head, tgt_len, head_dim]
            )
            # [src_len, batch, num_head * head_dim] -> [batch * num_head, src_len, head_dim]
            k = k.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim).transpose(0, 1)


        elif q.device.type == 'npu':
            q = (
                q.contiguous()
                .view(
                    tgt_len, bsz, self.num_head, self.head_dim
                )  # [tgt_len, batch * num_head, head_dim]
            )
            # [src_len, batch, num_head * head_dim] -> [batch * num_head, src_len, head_dim]
            k = k.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim)
            v = v.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim)
        
        # if self.use_rope and (self.rotary_embedding is not None):
        #     q,k= self.rotary_embedding(q,k)
        
        # self.dropout_p=0.0
        # out1= self.fwd_naiive(q,k,v, attention_mask)
        # out2= self.fwd_flash(q,k,v)
        # diff= out1-out2
        if (self.use_naiive or tgt_len != src_len) and q.device.type == 'npu':
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        use_mask = (attention_mask is not None) and tgt_len == src_len
        output= self.fwd_naiive(q,k,v,attention_mask)
        
        return output
       
    def fwd_naiive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        atttion_mask=None,
        use_mask: bool = True,
        fullmask: Optional[torch.BoolTensor] = None,
        seq_lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        nhead, hdim = q.shape[-2:]
        bsz = q.shape[0]
        tgt_len, src_len = q.shape[1], k.shape[1]
        # BTHD- >(BH)TD
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, tgt_len, hdim)
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, src_len, hdim)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, src_len, hdim)
        
        q = q.float() * self.norm_factor
        k = k.float()
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # print(torch.max(attn_weights), torch.min(attn_weights), attn_weights.requires_grad)
        if atttion_mask is not None:
            attn_weights = attn_weights + (atttion_mask.permute(2,0,1)-1)*1e4

        attention_probs = F.softmax(attn_weights, dim=-1).to(v)
        
        context = torch.bmm(attention_probs, v)

        output = context.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return output



class Window_CoreAttention(FlashCoreAttention):
    full_mask_cached: Dict[Tuple[int, torch.device], torch.Tensor] = dict()

    def __init__(
        self,
        head_dim: int,
        num_head: int,
        max_seq_length:int=4096,
        num_kv_head: Optional[int]=None,
        position_type="rope",
        dtype= torch.float16,
        attention_dropout_p=0.1,
        use_fa_v2=False,
        **kwargs
    ) -> None:
        super(Window_CoreAttention, self).__init__(head_dim=head_dim, num_head=num_head, max_seq_length=max_seq_length,
                                                        num_kv_head=num_kv_head, position_type=position_type,
                                                        dtype=dtype, attention_dropout_p=attention_dropout_p, use_fa_v2=use_fa_v2)
        self.use_naiive = True
        # logger.info('Window_CoreAttention does not support Flash Attention')
        
    def forward(
        self,
        q, k, v,
        padding_mask,
        attention_mask,
        relative_position_bias
    ):

        src_len, tgt_len = k.shape[0], q.shape[0]
        bsz = q.shape[1]
        if q.device.type == 'cuda':
            assert self.use_naiive, f"`Window_Attention` of Swin don't support `Flash Attention`"
            q = (
                q.contiguous()
                .view(
                    tgt_len, bsz, self.num_head, self.head_dim
                )  # [tgt_len, batch * num_head, head_dim]
                .transpose(0, 1)  # [batch * num_head, tgt_len, head_dim]
            )
            k = k.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim).transpose(0, 1)

            if q.shape[1] == 1:
                assert seq_lengths is None, f"`fwd_onestep` don't support `seq_lengths`"
                return self.fwd_onestep(q, k, v)
        elif q.device.type == 'npu':
            self.use_fa_v2 = False

            q = (
                q.contiguous()
                .view(
                    tgt_len, bsz, self.num_head, self.head_dim
                )  # [tgt_len, batch * num_head, head_dim]
            )
            # [src_len, batch, num_head * head_dim] -> [batch * num_head, src_len, head_dim]
            k = k.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim)
            v = v.contiguous().view(-1, bsz, self.num_kv_head, self.head_dim)
        
        if self.use_rope:
            q,k= self.rotary_embedding(q,k)
        if self.use_naiive and q.device.type == 'npu':
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        output= self.fwd_naiive(q,k,v, attention_mask, relative_position_bias,use_mask=False)  
        return output

    def fwd_naiive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask, 
        relative_position_bias,
        use_mask: bool = False,
        fullmask: Optional[torch.BoolTensor] = None,
        seq_lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        nhead, hdim = q.shape[-2:]
        bsz = q.shape[0]
        N = q.shape[1]
        tgt_len, src_len = q.shape[1], k.shape[1]
        # BTHD- >(BH)TD
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        q = q.float() * self.norm_factor
        k = k.float()
        attn_weights = (q @ k.transpose(-2, -1))
        attn_weights = attn_weights + relative_position_bias
        if attention_mask is not None:
            num_win = attention_mask.shape[0]
            attn_weights = attn_weights.view(bsz // num_win, num_win, self.num_head, N, N) + attention_mask.unsqueeze(1).unsqueeze(0)
            attn_weights = attn_weights.view(-1, self.num_head, tgt_len, src_len)

        attention_probs = F.softmax(attn_weights, dim=-1).to(v)
        
        context = (attention_probs @ v).transpose(1, 2).contiguous()
        context = context.view(bsz, tgt_len, -1)
        return context