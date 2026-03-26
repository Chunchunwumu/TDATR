import os
from typing import Optional
from functools import partial
from dataclasses import dataclass, field
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
from torch import Tensor
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.utils import init_tensor_parallel_parameters

from TDATR_utils.global_variables import ChoiceEnum
from typing import Optional, Callable, Tuple, List, Union, Optional
from TDATR.models.modules.linear_layer import ColumnParallelLinear, RowParallelLinear
from TDATR_utils.utils import nn_conv_weight_init_, nn_conv_bias_init_
from TDATR_utils.utils import divide
import logging
import warnings
import inspect
from TDATR.models.modules.mappings import copy_to_tensor_model_parallel_region, gather_from_tensor_model_parallel_region
from TDATR_utils.utils import initialize_weight_cpu, initialize_weight_gpu
logger= logging.getLogger(__name__)








def _is_tensor_parallel(partition_dim: int) -> bool:
    if partition_dim is None:
        return False
    if gpc.get_world_size(ParallelMode.TENSOR) <= 1:
        return False
    return True


class _OutChannelParallelConv(nn.Module):
    """
    Split the convolution kernel according to out_channel, where groups are fixed to 1
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 groups=1,
                 padding=0,
                 stride=1,
                 bias=True,
                 partial_stride=1,
                 weight_init_method=nn_conv_weight_init_,
                 bias_init_method=nn_conv_bias_init_,
                 gather_output=True,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(_OutChannelParallelConv, self).__init__()
        if groups != 1:
            raise ValueError('OutChannel-baed parallel conv only support groups = 1')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight_init_method = weight_init_method
        self.bias_init_method = bias_init_method
        self.gather_output = gather_output
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.skip_bias_add = skip_bias_add
        self.partial_stride = partial_stride
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        assert (out_channels // world_size) >= 1, "out_channels must be an integer multiple of world_size"
        self.out_channels_per_partition = divide(out_channels, world_size)

        cfgs = gpc.config
        if cfgs.common.fp16:
            params_dtype = torch.float16
        elif cfgs.common.bf16:
            params_dtype = torch.bfloat16
        else:
            params_dtype = torch.float32
        self.use_cpu_initialization = cfgs.model_parallel.use_cpu_initialization

        if cfgs.model_parallel.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.out_channels_per_partition,
                                                self.in_channels,
                                                *kernel_size,
                                                dtype=params_dtype))
        else:
            self.weight = Parameter(torch.empty(self.out_channels_per_partition,
                                                self.in_channels,
                                                *kernel_size,
                                                device=torch.cuda.current_device(),
                                                dtype=params_dtype))
        
        if bias:
            if cfgs.model_parallel.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.out_channels_per_partition, 
                                                dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(self.out_channels_per_partition, 
                                                device=torch.cuda.current_device(),
                                                dtype=params_dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if inspect.signature(self.bias_init_method).parameters.get("weight") is not None:
            self.bias_init_method = partial(self.bias_init_method, weight=self.weight)

        if self.use_cpu_initialization:
            initialize_weight_cpu(
                self.weight, 
                partition_dim=0, 
                per_partition_size=self.out_channels_per_partition,
                init_method=self.weight_init_method, 
                partition_stride=self.partial_stride,
            )
            if self.bias is not None:
                initialize_weight_cpu(
                    self.bias, 
                    partition_dim=0, 
                    per_partition_size=self.out_channels_per_partition,
                    init_method=self.bias_init_method, 
                    partition_stride=self.partial_stride,
                )
        else:
            initialize_weight_gpu(self.weight, self.weight_init_method, partition_dim=0)
            if self.bias is not None:
                initialize_weight_gpu(self.bias, self.bias_init_method, partition_dim=0)
            # for test
            # self.master_weight = gather_from_tensor_model_parallel_region(self.weight, dim=0)
            # self.master_bias = gather_from_tensor_model_parallel_region(self.bias, dim=0)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]=None):
        raise NotImplementedError("Please use OutChannelParallelConv1D, OutChannelParallelConv2D \
                                    or OutChannelParallelConv3D")

    def forward(self, input_):
        # Set up backprop all-reduce
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        
        # Convolution
        bias = self.bias if not self.skip_bias_add else None

        output_parallel = self._conv_forward(input_parallel, self.weight, bias)
        
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel, dim=1).contiguous()
        else:
            output = output_parallel
        
        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class OutChannelParallelConv2D(_OutChannelParallelConv):
    """
    Split the convolution kernel according to out_channel, where groups are fixed to 1
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding = 0,
                 stride=1,
                 bias=True,
                 partial_stride=1,
                 weight_init_method=nn_conv_weight_init_,
                 bias_init_method=nn_conv_bias_init_,
                 gather_output=True,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        elif isinstance(kernel_size, (Tuple, List)):
            assert len(kernel_size) == 2
        else:
            raise ValueError('unsupported kernel_size')

        super(OutChannelParallelConv2D, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = padding,
            stride = stride,
            bias = bias,
            partial_stride = partial_stride,
            weight_init_method = weight_init_method,
            bias_init_method = bias_init_method,
            gather_output = gather_output,
            keep_master_weight_for_test = keep_master_weight_for_test,
            skip_bias_add = skip_bias_add
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return F.conv2d(input, weight, bias, stride=self.stride, padding=self.padding)

