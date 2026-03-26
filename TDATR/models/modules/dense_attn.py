from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import flash_attn

from .xpos import XPOS,RotaryPositionalTransform
from TDATR_utils.global_context import global_context as gpc
import logging 
logger= logging.getLogger(__name__)

class FlashCoreAttention(nn.Module):
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
        super(FlashCoreAttention, self).__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.num_kv_head = num_kv_head if num_kv_head is not None else num_head
        self.dtype=dtype
        self.max_seq_length=max_seq_length
        self.use_rope= position_type != "none"
        self.use_fa_v2 = use_fa_v2

        self.rotary_embedding= None
        if position_type== "rope":
            self.rotary_embedding= RotaryPositionalTransform(self.head_dim, dtype=dtype)
        elif position_type =="xpos":
            self.rotary_embedding = XPOS(self.head_dim,scale_base=max_seq_length, dtype=dtype)
        if gpc.config.common.npu:
            self.use_naiive = True
        else:
            self.use_naiive = False
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

    def forward(
        self,
        q, k, v,
        padding_mask,
        attention_mask,
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

            if q.shape[1] == 1:
                assert seq_lengths is None, f"`fwd_onestep` don't support `seq_lengths`"
                return self.fwd_onestep(q, k, v)
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
        
        if self.use_rope:
            q,k= self.rotary_embedding(q,k)
        if self.use_naiive and q.device.type == 'npu':
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        output= self.fwd_naiive(q,k,v,use_mask=False)  
        return output

       
    def fwd_naiive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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
        if use_mask:
            if fullmask is None:
                if seq_lengths is None:
                    fullmask = self.get_fullmask(q.device)
                    fullmask = fullmask[-tgt_len:, -src_len:]
                else:
                    fullmask = build_local_sparse_mask(
                        seq_lengths, self.local_size, src_len, smooth=self.use_smooth
                    ).to(q.device)
                    fullmask = fullmask.expand(
                        bsz, self.num_head, src_len, src_len
                    ).reshape(-1, src_len, src_len)
                    fullmask = fullmask[:, -tgt_len:, -src_len:]
            else:
                tgt_len, src_len = fullmask.size()[-2:]
                fullmask = fullmask.expand(
                    bsz, self.num_head, tgt_len, src_len
                ).reshape(-1, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill_(fullmask, -10000.0)

        attention_probs = F.softmax(attn_weights, dim=-1).to(v)
        
        context = torch.bmm(attention_probs, v)

        output = context.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return output
