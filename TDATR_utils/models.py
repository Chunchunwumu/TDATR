"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

from typing import Optional, Union, Tuple
import numbers
import importlib

import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.cuda.amp import custom_fwd, custom_bwd
from TDATR.models.modules.linear_layer import LinearWithAsyncCommunication
from TDATR.models.modules.mappings import copy_to_tensor_model_parallel_region, gather_from_tensor_model_parallel_region

from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode

global hulk_layer_norm_cuda
hulk_layer_norm_cuda = None


class FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = hulk_layer_norm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = hulk_layer_norm_cuda.backward_affine(
            grad_output.contiguous(),
            mean,
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            bias_,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None, None


class MixedFusedLayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int]],
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        sequence_parallel: Optional[bool] = None,
    ):
        super(MixedFusedLayerNorm, self).__init__()

        global hulk_layer_norm_cuda
        if hulk_layer_norm_cuda is None:
            try:
                hulk_layer_norm_cuda = importlib.import_module(
                    "TDATR_utils.layer_norm_cuda"
                )
            except ImportError as e:
                raise RuntimeError(
                    f"import `MixedFusedLayerNorm`(cuda extensions) error: {e}."
                )

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.apply_layernorm_1p = gpc.config.common.apply_layernorm_1p

        self.weight = Parameter(
            torch.empty(*normalized_shape, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            torch.empty(*normalized_shape, device=device, dtype=dtype)
        )

        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel is None:
            self.sequence_parallel = gpc.config.model_parallel.sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        setattr(self.bias, "sequence_parallel", self.sequence_parallel)
        self.reset_parameters()

    def reset_parameters(self):
        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

        outputs = FusedLayerNormAffineFunction.apply(
            input.to(self.weight),
            weight,
            self.bias,
            self.normalized_shape,
            self.eps,
        )
        
        return outputs.to(input)

    def __repr__(self):
        return f"MixedFusedLayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"

"""This code from NVIDIA Megatron
   with some changes. """

import enum

import torch
import torch.nn as nn




def parallel_lm_logits(
    inputs_parallel,
    word_embeddings_weight,
    parallel_output,
):
    
    # [b, s, h] -> [s, b, h]
    inputs_parallel = inputs_parallel.transpose(0, 1).contiguous()

    async_tensor_model_parallel_allreduce = \
        gpc.config.model_parallel.async_tensor_model_parallel_allreduce
    sequence_parallel = gpc.config.model_parallel.sequence_parallel

    if async_tensor_model_parallel_allreduce or sequence_parallel:
        inputs_parallel = inputs_parallel
        tensor_parallel = gpc.get_world_size(ParallelMode.TENSOR) > 1
        async_grad_allreduce = async_tensor_model_parallel_allreduce and \
                               tensor_parallel and not sequence_parallel
    else:
        inputs_parallel = copy_to_tensor_model_parallel_region(inputs_parallel)
        async_grad_allreduce = False

    logits_parallel = LinearWithAsyncCommunication.apply(
        inputs_parallel,
        word_embeddings_weight,
        None,
        async_grad_allreduce,
        sequence_parallel,
    )

    # [s, b, h] -> [b, s, h]
    logits_parallel = logits_parallel.transpose(0, 1).contiguous()

    if parallel_output:
        return logits_parallel
    
    return gather_from_tensor_model_parallel_region(logits_parallel)