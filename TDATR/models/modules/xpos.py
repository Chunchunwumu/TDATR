# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
from typing import List, Dict, Tuple
import math
import numpy as np
import torch
import torch.nn as nn

# all input shape is B,T,H,D


@torch.jit.script
def get_emd(q, q_o, cos_q, sin_q):
    return q * cos_q + q_o * sin_q


@torch.jit.script
def get_sin_cos(seq_len: int, inv_freq, dtype: torch.dtype):
    seq_len = torch.jit.annotate(int, seq_len)
    freqs = inv_freq * (seq_len - 1)
    cos_t = freqs.cos()[None, None, None, :].to(dtype)
    sin_t = freqs.sin()[None, None, None, :].to(dtype)
    cos_t = torch.cat((cos_t, cos_t), dim=-1)
    sin_t = torch.cat((sin_t, sin_t), dim=-1)
    return cos_t, sin_t


class RotaryPositionalTransform(torch.nn.Module):
    cos_sin_cached: Dict[
        Tuple[int, int, int, torch.dtype, torch.device],
        Tuple[torch.Tensor, torch.Tensor],
    ] = dict()

    def __init__(self, dim: int, base: int = 10000, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.dtype: torch.dtype = dtype
        self.base: int = base
        self.dim: int = dim

    def get_cos_sin(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input x with B XT X  H X D
            seq_len: Sequence length of input x
        """
        device = x.device
        if device.type == "npu":
            seq_len = x.size(0)
            pad_seq_len = int(math.ceil(x.size(0) / 1024) * 1024)
        else:
            seq_len = x.size(1)
            pad_seq_len = int(math.ceil(x.size(1) / 1024) * 1024)
        cache_key = (pad_seq_len, self.base, self.dim, self.dtype, x.device)

        if cache_key not in RotaryPositionalTransform.cos_sin_cached:
            # these initialization must be in float32
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
            )
            inv_freq = inv_freq.to(device)
            t = torch.arange(pad_seq_len, dtype=torch.float, device=device)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            if device.type == "npu":
                cos_cached = emb.cos()[:, None, None, :].to(dtype=self.dtype, device=device)
                sin_cached = emb.sin()[:, None, None, :].to(dtype=self.dtype, device=device)
            else:
                cos_cached = emb.cos()[None, :, None, :].to(dtype=self.dtype, device=device)
                sin_cached = emb.sin()[None, :, None, :].to(dtype=self.dtype, device=device)
            RotaryPositionalTransform.cos_sin_cached[cache_key] = (
                cos_cached,
                sin_cached,
            )

        cos_cached, sin_cached = RotaryPositionalTransform.cos_sin_cached[cache_key]
        if device.type == "npu":
            return cos_cached[:seq_len], sin_cached[:seq_len]
        else:
            return cos_cached[:, :seq_len], sin_cached[:, :seq_len]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q: B X T_q  X  H X D
        k: B X T X  H X D
        while training, T_q=T, while inference, T_q=1
        """

        cos, sin = self.get_cos_sin(k)

        if q.device.type == "npu":
            if q.shape[0] > k.shape[0]:
                raise ValueError(f"q shape {q.shape[0]} bigger than k {k.shape[0]}")
            if q.shape[0] < k.shape[0]:
                cos_q, sin_q = cos[-q.shape[0]:], sin[-q.shape[0]:]
            else:
                cos_q, sin_q = cos, sin
        else:
            if q.shape[1] > k.shape[1]:
                raise ValueError(f"q shape {q.shape[1]} bigger than k {k.shape[1]}")
            if q.shape[1] < k.shape[1]:
                cos_q, sin_q = cos[:, -q.shape[1] :], sin[:, -q.shape[1] :]
            else:
                cos_q, sin_q = cos, sin
        q = q * cos_q + self.rotate_half(q) * sin_q
        k = k * cos + self.rotate_half(k) * sin
        return q, k

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class XPOS(torch.nn.Module):
    def __init__(self, dim, base=10000, scale_base=512, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype
        self.base = base
        self.dim = dim
        self.scale_base = scale_base
        # inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq=inv_freq
        # self.register_buffer("inv_freq", inv_freq)
        # self.register_buffer(
        #     "scale", (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        # )
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.scale_cached = None

    def get_cos_sin(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input x with B X T X H X C
            seq_len: Sequence length of input x
        """
        if x.device.type == "npu":
            seq_len = x.shape[0]
        else:
            seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            # these initialization must be in float32
            self.seq_len_cached = seq_len
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
            )
            inv_freq = inv_freq.to(x.device)
            t = torch.arange(seq_len, device=x.device).float()
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, :, None, :].to(self.dtype)
            self.sin_cached = emb.sin()[None, :, None, :].to(self.dtype)
            min_pos = -(seq_len + offset) // 2
            max_pos = seq_len + offset + min_pos
            scale = (torch.arange(0, self.dim, 2) + 0.4 * self.dim) / (1.4 * self.dim)
            scale = scale.to(x.device)
            scale = (
                scale
                ** torch.arange(min_pos, max_pos, 1)
                .to(scale)
                .div(self.scale_base)[:, None]
            )
            self.scale_cached = torch.cat((scale, scale), dim=-1).to(self.dtype)
            self.scale_cached = self.scale_cached[None, :, None, :]
        return self.cos_cached, self.sin_cached, self.scale_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q: B X T_q   X H X C
        k: B X T X   X H X C
        while training, T_q=T, while inference, T_q=1
        """

        cos, sin, scale = self.get_cos_sin(k)
        if q.device.type == "npu":
            if q.shape[0] > k.shape[0]:
                raise ValueError(f"q shape {q.shape[0]} bigger than k {k.shape[0]}")
            scale_q = scale
            scale_k = 1 / scale
            if q.shape[0] < k.shape[0]:
                cos_q, sin_q = cos[:, -q.shape[0] :], sin[:, -q.shape[0] :]
                scale_q = scale_q[:, -q.shape[0] :]
            else:
                cos_q, sin_q = cos, sin
        else:
            if q.shape[1] > k.shape[1]:
                raise ValueError(f"q shape {q.shape[1]} bigger than k {k.shape[1]}")
            scale_q = scale
            scale_k = 1 / scale
            if q.shape[1] < k.shape[1]:
                cos_q, sin_q = cos[:, -q.shape[1] :], sin[:, -q.shape[1] :]
                scale_q = scale_q[:, -q.shape[1] :]
            else:
                cos_q, sin_q = cos, sin
        q = (q * cos_q + self.rotate_half(q) * sin_q) * scale_q
        k = (k * cos + self.rotate_half(k) * sin) * scale_k
        # if torch.isnan(q.mean()) or torch.isnan(k.mean()):
        #     print("hello world")
        return q, k

    # rotary pos emb helpers:
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
