from typing import List, Tuple, Dict, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F
import einops
import warnings
from .xpos import XPOS, RotaryPositionalTransform

from TDATR_utils.global_context import global_context as gpc

import warnings
from einops import rearrange, repeat
import logging
logger = logging.getLogger(__name__)
EPSILON = 1e-6


def get_blockmask(seqlen, local_size=256, blocksize_q=16, blocksize_k=128):
    assert (
        seqlen % blocksize_k == 0
    ), f"seqlen {seqlen} not dividable by blocksize_k{blocksize_k}"
    assert (
        blocksize_k % blocksize_q == 0
    ), f"k_block{blocksize_k} should be times of blocksize_q{blocksize_q}"
    qlen, klen = seqlen // blocksize_q, seqlen // blocksize_k
    qrange = torch.arange(qlen)
    krange = torch.arange(klen)
    # (k_start_index  < q_start_index -local_size) or (k_start_index >q_start_index+ q_bucket_size)
    # import pdb
    # mask1=  krange[None,:]*blocksize_k>= qrange[:,None]*blocksize_q+blocksize_q
    # mask2= krange[None,:]*blocksize_k <= qrange[:, None]*blocksize
    mask1 = qrange[:, None] * blocksize_q < krange[None, :] * blocksize_k
    mask2 = (
        qrange[:, None] * blocksize_q >= (krange[None, :]) * blocksize_k + local_size
    )
    mask = mask1 | mask2
    return mask


def expand_blockmask(
    layout: torch.Tensor, block_q: int = 16, block_k: int = 128
) -> torch.BoolTensor:
    qlen, klen = layout.shape
    seqlen = qlen * block_q
    assert seqlen == block_k * klen
    fullmask = layout.view(qlen, 1, klen, 1).repeat(1, block_q, 1, block_k)
    fullmask = fullmask.reshape(seqlen, seqlen)
    #  0 for good,1 for padding
    seqid = torch.arange(seqlen, dtype=torch.long)
    causal_mask = seqid[:, None] < seqid[None, :]
    fullmask = fullmask | causal_mask
    return fullmask


def get_smooth_local_mask(seqlen_q, seqlen_k, local_size, device):
    assert seqlen_k > 1 and seqlen_k > 0
    max_seq_len = max(seqlen_q, seqlen_k)
    mask1 = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
    mask2 = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=(1 - local_size)).bool()
    mask = ~(mask1 & mask2)
    mask = mask[-seqlen_q:, -seqlen_k:]
    mask = mask.to(device)
    return mask


class SparseSelfAttention(nn.Module):
    full_mask_cached: Dict[
        Tuple[int, int, int, int, torch.device], torch.Tensor
    ] = dict()

    def __init__(
        self,
        head_dim: int,
        num_head: int,
        num_kv_head: Optional[int]=None,
        max_seq_length=4096,
        local_size=1024,
        position_type="rope",
        rope_base: int = 10000,
        attention_dropout_p=0.1,
        dtype=torch.float16,
        use_naiive=False,
        use_smooth:bool=False,
        use_fa_v2=False,
        **kwargs,
    ) -> None:
        super(SparseSelfAttention, self).__init__()
        assert position_type in [
            "rope",
            "xpos",
            "none",
        ], f"now only support rope, xpos, none"
        self.head_dim = head_dim
        self.num_head = num_head
        self.num_kv_head = num_kv_head if num_kv_head is not None else num_head


        self.use_rope = position_type != "none"
        self.rope_base: int = rope_base
        self.max_seq_length = max_seq_length
        self.rotary_embedding = None
        self.use_fa_v2 = use_fa_v2
        if position_type == "rope":
            self.rotary_embedding = RotaryPositionalTransform(
                self.head_dim, self.rope_base, dtype=dtype
            )
        elif position_type == "xpos":
            self.rotary_embedding = XPOS(
                self.head_dim, scale_base=local_size, dtype=dtype
            )

        self.use_naiive = use_naiive
        if torch.cuda.get_device_capability(0) is not None and torch.cuda.get_device_capability(0)[0] < 7.5:
            warnings.warn(
                "NOTE: your device does NOT support flash attention, back to naiive"
            )
            self.use_naiive = True
        self.use_smooth = use_smooth
        self.norm_factor = 1.0 / math.sqrt(head_dim)
        # cfg = gpc.config.model
        # self.dropout_p= cfg.attention_dropout_p
        self.dropout_p = attention_dropout_p

        self.block_q, self.block_k = 16, 128
        self.local_size = local_size
        # layout= get_blockmask(max_seq_length,local_size, self.block_q, self.block_k)
        # self.register_buffer("layout", layout)
        # # transform to fairseq mask: 0 for normal and 1 for ignore
        # fullmask= self.expand_blockmask(layout)
        # self.register_buffer('fullmask', fullmask)

    def expand_blockmask(self, layout: torch.Tensor) -> torch.Tensor:
        qlen, klen = layout.shape
        seqlen = qlen * self.block_q
        assert seqlen == self.block_k * klen
        fullmask = layout.view(qlen, 1, klen, 1).repeat(
            1, self.block_q, 1, self.block_k
        )
        fullmask = fullmask.reshape(seqlen, seqlen)
        #  0 for good,1 for padding
        seqid = torch.arange(seqlen, dtype=torch.long)
        causal_mask = seqid[:, None] < seqid[None, :]
        removed = (~fullmask) & causal_mask

        fullmask = fullmask | causal_mask
        return fullmask

    def get_fullmask(self, device: torch.device, dtype=torch.bool) -> torch.BoolTensor:
        cache_key = (
            self.max_seq_length,
            self.local_size,
            self.block_q,
            self.block_k,
            device,
        )
        if cache_key not in SparseSelfAttention.full_mask_cached:
            if self.use_smooth:
                mask = get_smooth_local_mask(
                    seqlen_q=self.max_seq_length,
                    seqlen_k=self.max_seq_length,
                    local_size=self.local_size,
                    device=device
                )
            else:
                layout = get_blockmask(
                    self.max_seq_length, self.local_size, self.block_q, self.block_k
                )
                mask = self.expand_blockmask(layout).to(device)
            SparseSelfAttention.full_mask_cached[cache_key] = mask
        return SparseSelfAttention.full_mask_cached[cache_key]

    def decode_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(self.rotary_embedding, RotaryPositionalTransform)
        # 375a34cbff46017939bc1568b76ea6534bc308bc
        bsz = q.size(1)
        # T, B, H -> T, B, n, h -> B * n, T, h
        q = q.reshape(-1, bsz * self.num_head, self.head_dim).permute(1, 0, 2).contiguous()
        k = k.reshape(-1, bsz * self.num_kv_head, self.head_dim).permute(1, 0, 2).contiguous()
        v = v.reshape(-1, bsz * self.num_kv_head, self.head_dim).permute(1, 0, 2).contiguous()

        if self.use_rope:
            cos_q, sin_q = inference_params.get_rope_constants()
            q = q * cos_q + self.rotate_half(q) * sin_q
            k = k * cos_q + self.rotate_half(k) * sin_q

        if inference_params is not None:
            k, v = inference_params.update(inference_params.layer_idx, k, v)
        
        q = q.float() * self.norm_factor
        k = k.float()
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights.masked_fill_(inference_params.get_attention_mask(), -10000.0)

        attention_probs = F.softmax(attn_weights, dim=-1).to(v)
        context = torch.bmm(attention_probs, v)
        output = context.transpose(0, 1).contiguous().view(1, bsz, -1)
        return output

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def fwd_onestep(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        tlen, slen = q.shape[1], k.shape[1]
        assert tlen == 1, "fwd_onestep only for inference"

        if self.use_rope:
            q, k = self.rotary_embedding(q, k)

        if slen > self.local_size:
            s_begin = (
                slen + self.block_k - 1
            ) // self.block_k * self.block_k - self.local_size
            k = k[:, s_begin:]
            v = v[:, s_begin:]

        if self.use_fa_v2:
            output = self.fwd_naiive_v2(q, k, v, use_mask=False)
        else:
            output = self.fwd_naiive(q, k, v, use_mask=False)
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: Optional[torch.LongTensor] = None,
        inference_params = None,
        **kwargs,
    ) -> torch.Tensor:
        in_generate_engine = hasattr(inference_params, 'status')
        if in_generate_engine and inference_params.status.name == 'decode':
            return self.decode_forward(q, k, v, padding_mask, attention_mask, seq_lengths, inference_params, **kwargs)

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
            q, k = self.rotary_embedding(q, k)
        # if inference_params is not None:
        if in_generate_engine and inference_params.status.name == 'encode':
            k, v = inference_params.update(inference_params.layer_idx, k, v)
        
        if self.use_naiive and q.device.type == 'npu':
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        assert self.num_kv_head == self.num_head, "naive flash attention v1 has not yet supported GQA"
        output = self.fwd_naiive(q, k, v, seq_lengths=seq_lengths)
        
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
        attention_probs = F.dropout(
            attention_probs, p=self.dropout_p, training=self.training
        )
        context = torch.bmm(attention_probs, v)

        output = context.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        return output

    def _load_from_state_dict(
        self, state_dict: Dict, prefix: int, *args, **kwargs
    ) -> None:
        self.prefix = prefix
        ignore_buffers = {prefix + i for i in ["layout", "fullmask"]}
        param_names = list(state_dict.keys())
        for param_name in param_names:
            if param_name in ignore_buffers:
                state_dict.pop(param_name)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def build_local_sparse_mask(
    cumul_seq_lengths: torch.Tensor, local_size: int, max_len: int, smooth: bool = False,
) -> torch.BoolTensor:
    warnings.warn(
        "`build_local_sparse_mask` is only used for consistency alignment, "
        "and should not be used in training envorment as it is inefficient."
    )
    seq_lengths = cumul_seq_lengths[1:] - cumul_seq_lengths[:-1]
    max_sample_length = seq_lengths.max()
    max_sample_length = math.ceil(max_sample_length / 128) * 128
    if smooth:
        fullmask = get_smooth_local_mask(
            max_sample_length,
            max_sample_length,
            local_size,
            device=cumul_seq_lengths.device
        )
    else:
        layout = get_blockmask(max_sample_length, local_size=local_size)
        fullmask = expand_blockmask(layout)
    bsz = cumul_seq_lengths[-1] // max_len
    assert bsz * max_len == cumul_seq_lengths[-1]
    attention_mask = ~torch.ones((bsz, 1, max_len, max_len), dtype=torch.bool).tril()
    positions = [[] for _ in range(bsz)]
    for len in cumul_seq_lengths[1:]:
        len = int(len - 1)
        sample_id = len // max_len
        len = len % max_len
        positions[sample_id].append(len + 1)
    for b in range(bsz):
        prev_pos = 0
        for pos in positions[b]:
            attention_mask[b, 0, pos:, :pos] = True
            attention_mask[b, 0, prev_pos:pos, prev_pos:pos] |= fullmask[
                : pos - prev_pos, : pos - prev_pos
            ]
            prev_pos = pos
    return attention_mask


def build_decode_local_sparse_mask(
    cumul_seq_lengths: torch.Tensor, local_size: int
) -> torch.BoolTensor:
    warnings.warn(
        "`build_local_sparse_mask` is only used for consistency alignment, "
        "and should not be used in training envorment as it is inefficient."
    )
    cumul_seq_lengths = cumul_seq_lengths.cpu()
    max_len = cumul_seq_lengths[-1]
    seq_lengths = cumul_seq_lengths[1:] - cumul_seq_lengths[:-1]
    max_sample_length = seq_lengths.max()
    if max_sample_length > local_size+259:
        a = 2
    max_sample_length = math.ceil(max_sample_length / 128) * 128
    layout = get_blockmask(max_sample_length, local_size=local_size)
    fullmask = expand_blockmask(layout)
    attention_mask = ~torch.ones((1, 1, max_len, max_len), dtype=torch.bool).tril()
    prev_pos = cumul_seq_lengths[0]
    for pos in cumul_seq_lengths[1:]:
        attention_mask[0, 0, pos:, :pos] = True
        attention_mask[0, 0, prev_pos:pos, prev_pos:pos] |= fullmask[
                :pos - prev_pos, :pos - prev_pos
            ]
        prev_pos = pos
    attention_mask = attention_mask[:, :, cumul_seq_lengths[1:]-1, :]
    return attention_mask.cuda()
