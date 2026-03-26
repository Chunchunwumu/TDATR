# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import MultiheadAttention


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        # self_attn_cfg=dict(embed_dims=256, num_levels=4,
        #     dropout=0.0),  # 0.1 for DeformDETR

        self.self_attn = MultiheadAttention(d_model, nhead, attn_drop=attn_dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        # pass
        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.normal_(self.linear2.weight, std=0.01)

        # linear_init_(self.linear1)
        # linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(ch_out)
        if act is not None:
            self.act = nn.GELU()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        if self.act is not None:
            x = self.act(x)

        return x

class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight, gain=1)
        # nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.conv(x)
        
        # pdb.set_trace()
        x = self.bn(x)
        y = x * torch.sigmoid(x)
        return y

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', alpha=False,deploy=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)

        self.alpha = None
        self.act = act

        self.depoly = deploy

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

        for para in self.parameters():
            para.detach_()

        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1


    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])


    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        # import pdb
        # pdb.set_trace()
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)



class HybridEncoder(nn.Module):

    def __init__(self,
                 mask=False,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 num_heads=8,
                 dropout=0,
                 attn_dropout=None,
                 act_dropout=None,
                 feedforward_dim=1024,
                 act='silu',
                #  trt=False,
                #  batch_first=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.embed_dims = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size
        self.mask = mask
        # channel projection

        # self.input_proj = nn.ModuleList()
        # for in_channel in in_channels:
        #     self.input_proj.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
        #             nn.BatchNorm2d(hidden_dim)))

        # encoder transformer
        self.encoder = nn.ModuleList([
            TransformerLayer(
                 d_model=hidden_dim,
                 nhead=num_heads,
                 dim_feedforward=feedforward_dim,
                 dropout=dropout,
                 activation="relu",
                 attn_dropout=attn_dropout,
                 act_dropout=act_dropout,
                 normalize_before=False)
            for _ in range(len(use_encoder_idx))
        ])

        act = nn.GELU()
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return torch.cat(
            [
                torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),
                torch.cos(out_h)
            ],
            dim=1)[None, :, :]

    def _get_encoder_input(self, feats):
        # get projection features
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # if self.num_feature_levels > len(proj_feats):
        #     len_srcs = len(proj_feats)
        #     for i in range(len_srcs, self.num_feature_levels):
        #         if i == len_srcs:
        #             proj_feats.append(self.input_proj[i](feats[-1]))
        #         else:
        #             proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        # spatial_shapes = []
        # level_start_index = [0, ]
        proj_feats = feats
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            # if self.mask:
            #     if i in [1,2]:
            #         continue
            # else:
            #     if i in [0,1]:
            #         continue
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1).contiguous())
            # [num_levels, 2]
            # spatial_shapes.append([h, w])
            # [l], start index of each level
            # level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        # level_start_index.pop()
        return feat_flatten
        # return (feat_flatten, spatial_shapes, level_start_index)

    def forward(self, feats, multi_level_pos_embeds):
        
        # pdb.set_trace()
        assert len(feats) == len(self.in_channels)
        # get projection features
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        proj_feats = list(feats)
        # encoder
        # import time
        # torch.cuda.synchronize()
        # s= time.time()
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1).contiguous()
                pos_embed = multi_level_pos_embeds[enc_ind].flatten(2).permute(0, 2, 1).contiguous()              

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).contiguous().reshape(-1, self.hidden_dim, h, w)
                # print('pos embed+ 1 encoder:',time.time()-sss)

        # torch.cuda.synchronize()
        # ss = time.time()
        # print('pos embed+ 1 encoder:',ss - s)
        # pdb.set_trace()
        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # torch.cuda.synchronize()
        # sss = time.time()
        # print('top down:',sss - ss)
        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)
        # torch.cuda.synchronize()
        # print('bottom up:', time.time() - sss)

        memory = self._get_encoder_input(outs)
        return memory
