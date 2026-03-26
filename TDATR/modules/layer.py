import os
from typing import Optional
from functools import partial
from dataclasses import dataclass, field
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from TDATR_utils.global_context import global_context as gpc
import inspect
 
from typing import Optional, Callable, Tuple
from TDATR.models.modules.linear_layer import ColumnParallelLinear, RowParallelLinear

import logging
import warnings

logger= logging.getLogger(__name__)


class ModelParallelMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_embed_dim: int,
        dtype:Optional[torch.dtype]=torch.float16,
        skip_last_bias_add: bool=True,
        gate_gelu= False
    ) -> None:
        super(ModelParallelMLP, self).__init__()
        cfg = gpc.config
        self.cfg = cfg
        self.use_cpu_initialization = cfg.model_parallel.use_cpu_initialization
        self.embed_dim = embed_dim
        self.mlp_embed_dim = mlp_embed_dim
        self.dtype = dtype
        self.gate_gelu = gate_gelu

        self.fc1 = self.build_fc1(
            gather_output=False,
            skip_bias_add=False,
        )
        self.activation_fn = self.build_activation()
        self.dropout = self.build_dropout()
        self.fc2 = self.build_fc2(
            input_is_parallel=True,
            skip_bias_add=skip_last_bias_add,
        )
    
    def build_fc1(
        self, 
        gather_output: Optional[bool]=False,
        init_method: Optional[Callable]=init.xavier_normal_,
        skip_bias_add: Optional[bool]=False,
    ):
        return ColumnParallelLinear(
            self.embed_dim,
            self.mlp_embed_dim,
            gather_output=gather_output,
            dtype=self.dtype,
            weight_init_method=init_method,
            use_cpu_initialization=self.use_cpu_initialization,
            skip_bias_add=skip_bias_add
        )

    def build_fc2(
        self, 
        input_is_parallel: Optional[bool]=False,
        init_method: Optional[Callable]=init.xavier_normal_,
        skip_bias_add: Optional[bool]=False,
    ):
        if self.gate_gelu:
            return RowParallelLinear(
                self.mlp_embed_dim // 2,
                self.embed_dim,
                input_is_parallel=input_is_parallel,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                weight_init_method=init_method,
                skip_bias_add=skip_bias_add
            )
        else:
            return RowParallelLinear(
                self.mlp_embed_dim,
                self.embed_dim,
                input_is_parallel=input_is_parallel,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                weight_init_method=init_method,
                skip_bias_add=skip_bias_add
            )
        
    def build_activation(
        self,
    ):
        return nn.GELU()
    
    def build_dropout(
        self,
        p = 0.0
    ):
        return None
    
    def forward(self, inputs):
        def fc1_linear(inputs):
            intermediate_parallel, _ = self.fc1(inputs)
            return intermediate_parallel
        def gate_gelu(intermediate_parallel):
            if self.gate_gelu:
                # fix bug: chunk may cause incomformity between different tensor parallel
                hshape= intermediate_parallel.shape[:-1]
                intermediate_parallel= intermediate_parallel.view(hshape+(-1,2))
                #intermediate_parallel1,intermediate_parallel2= intermediate_parallel[...,0],intermediate_parallel[...,1]
                intermediate_parallel1, intermediate_parallel2 = torch.chunk(intermediate_parallel, 2, dim=-1)
                intermediate_parallel1 = intermediate_parallel1.squeeze(-1)
                intermediate_parallel2 = intermediate_parallel2.squeeze(-1)
                # set_trace()
                intermediate_parallel1 = self.activation_fn(intermediate_parallel1)
                intermediate_parallel = intermediate_parallel1 * intermediate_parallel2
                intermediate_parallel = intermediate_parallel.clamp(-32.0, 32.0)
                
            else:
                intermediate_parallel = self.activation_fn(intermediate_parallel)
                intermediate_parallel = intermediate_parallel.clamp(-32.0, 32.0)
            return intermediate_parallel

        def fc1_linear_gate_gelu(inputs):
            intermediate_parallel = fc1_linear(inputs)
            intermediate_parallel = gate_gelu(intermediate_parallel)
            return intermediate_parallel

        intermediate_parallel = fc1_linear_gate_gelu(inputs)

        output, output_bias = self.fc2(intermediate_parallel)
        return output, output_bias


class ModelParallelMLPDeprec(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_embed_dim: int,
        dtype:Optional[torch.dtype]=torch.float16,
        skip_last_bias_add: bool=True,
        gate_gelu= False
    ) -> None:
        super(ModelParallelMLPDeprec, self).__init__()
        warnings.warn("NOTE: You use ModelParallelMLPDeprec which is deprecated and only for ipt v3 or earlier. this version is in trouble with tensor parallel")
        cfg = gpc.config
        self.use_cpu_initialization = cfg.model_parallel.use_cpu_initialization
        self.embed_dim = embed_dim
        self.mlp_embed_dim = mlp_embed_dim
        self.dtype = dtype
        self.gate_gelu = gate_gelu

        self.fc1 = self.build_fc1(
            gather_output=False,
            skip_bias_add=False,
        )
        self.activation_fn = self.build_activation()
        self.dropout = self.build_dropout()
        self.fc2 = self.build_fc2(
            input_is_parallel=True,
            skip_bias_add=skip_last_bias_add,
        )
    
    def build_fc1(
        self, 
        gather_output: Optional[bool]=False,
        init_method: Optional[Callable]=init.xavier_normal_,
        skip_bias_add: Optional[bool]=False,
    ):
        return ColumnParallelLinear(
            self.embed_dim,
            self.mlp_embed_dim,
            gather_output=gather_output,
            dtype=self.dtype,
            weight_init_method=init_method,
            use_cpu_initialization=self.use_cpu_initialization,
            skip_bias_add=skip_bias_add
        )

    def build_fc2(
        self, 
        input_is_parallel: Optional[bool]=False,
        init_method: Optional[Callable]=init.xavier_normal_,
        skip_bias_add: Optional[bool]=False,
    ):
        if self.gate_gelu:
            return RowParallelLinear(
                self.mlp_embed_dim // 2,
                self.embed_dim,
                input_is_parallel=input_is_parallel,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                weight_init_method=init_method,
                skip_bias_add=skip_bias_add
            )
        else:
            return RowParallelLinear(
                self.mlp_embed_dim,
                self.embed_dim,
                input_is_parallel=input_is_parallel,
                dtype=self.dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                weight_init_method=init_method,
                skip_bias_add=skip_bias_add
            )
        
    def build_activation(
        self,
    ):
        return nn.GELU()
    
    def build_dropout(
        self,
        p = 0.0
    ):
        return None
    
    def forward(self, inputs):
        intermediate_parallel, _ = self.fc1(inputs)
        if self.gate_gelu:
            # fix bug: chunk may cause incomformity between different tensor parallel
            # hshape= intermediate_parallel.shape[:-1]
            # intermediate_parallel= intermediate_parallel.view(hshape+(-1,2))
            # intermediate_parallel1,intermediate_parallel2= intermediate_parallel[...,0],intermediate_parallel[...,1]
            intermediate_parallel1, intermediate_parallel2 = torch.chunk(intermediate_parallel, 2, dim=-1)
            # set_trace()
            intermediate_parallel1 = self.activation_fn(intermediate_parallel1)
            intermediate_parallel = intermediate_parallel1 * intermediate_parallel2
            intermediate_parallel = intermediate_parallel.clamp(-32.0, 32.0)
        else:
            intermediate_parallel = self.activation_fn(intermediate_parallel)
            intermediate_parallel = intermediate_parallel.clamp(-32.0, 32.0)
        output, output_bias = self.fc2(intermediate_parallel)
        
        return output, output_bias

