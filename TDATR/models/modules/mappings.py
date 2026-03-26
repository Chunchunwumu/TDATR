# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from TDATR_utils.utils import split_tensor
# from TDATR.models.modules.transformer_layer_effiency import 

from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.global_context import global_context as gpc



def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if gpc.get_world_size(ParallelMode.TENSOR) == 1:
        return input_

    # All-reduce.
    tensor_model_parallel_group = gpc.get_group(ParallelMode.TENSOR)
    torch.distributed.all_reduce(input_, group=tensor_model_parallel_group)

    return input_


def _split(input_, dim=-1):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = gpc.get_world_size(ParallelMode.TENSOR)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along the dim specified dimension.
    input_list = split_tensor(input_, world_size, dim)

    # Note: torch.split does not create contiguous tensors by default.
    rank = gpc.get_local_rank(ParallelMode.TENSOR)
    output = input_list[rank].contiguous()

    return output


def _gather(input_, dim=-1):
    """Gather tensors and concatinate along the last dimension."""

    world_size = gpc.get_world_size(ParallelMode.TENSOR)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    rank = gpc.get_local_rank(ParallelMode.TENSOR)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    tensor_model_parallel_group = gpc.get_group(ParallelMode.TENSOR)
    torch.distributed.all_gather(tensor_list, input_, group=tensor_model_parallel_group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _reduce_scatter(input_, dim=-1):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = gpc.get_world_size(ParallelMode.TENSOR)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    
    dim_size[0] = dim_size[0] // world_size
   
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    if hasattr(torch.distributed, "reduce_scatter_tensor"):
        torch.distributed.reduce_scatter_tensor(
            output, 
            input_.contiguous(), 
            group=gpc.get_group(ParallelMode.TENSOR)
        )
    else:
        torch.distributed._reduce_scatter_base(
            output, 
            input_.contiguous(), 
            group=gpc.get_group(ParallelMode.TENSOR)
        )
    return output

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim=-1):
        return _split(input_, dim=dim)

    @staticmethod
    def forward(ctx, input_, dim=-1):
        ctx.dim = dim
        return _split(input_, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        return _gather(grad_output, dim=dim), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim=-1):
        return _gather(input_, dim=dim)
    
    @staticmethod
    def forward(ctx, input_, dim=-1):
        ctx.dim = dim
        return _gather(input_, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        return _split(grad_output, dim=dim), None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim=0):
        return _split(input_, dim=dim)

    @staticmethod
    def forward(ctx, input_, dim=0):
        ctx.dim = dim
        return _split(input_, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        return _gather(grad_output, dim=dim), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate.""" 

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True, dim=0):
        return _gather(input_, dim=dim)
    
    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True, dim=0):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.dim = dim
        return _gather(input_, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad
        dim = ctx.dim
        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce 
        # scattered and whereas if the computation is duplicated, 
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter(grad_output, dim=dim), None, None
        else:
            return _split(grad_output, dim=dim), None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, dim=0):
        return _reduce_scatter(input_, dim=dim)
    
    @staticmethod
    def forward(ctx, input_, dim=0):
        ctx.dim = dim
        return _reduce_scatter(input_, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        return _gather(grad_output, dim=dim)


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_, dim=-1):
    return _ScatterToModelParallelRegion.apply(input_, dim)


def gather_from_tensor_model_parallel_region(input_, dim=-1):
    return _GatherFromModelParallelRegion.apply(input_, dim)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True, dim=0):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad, dim)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)
