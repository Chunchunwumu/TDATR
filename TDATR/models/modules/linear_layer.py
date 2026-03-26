from typing import Optional, Callable
import inspect
from functools import partial

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_context import global_memory_buffer
from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.utils import VocabUtility, divide, initialize_weight_cpu, initialize_weight_gpu

from TDATR.models.modules.mappings import copy_to_tensor_model_parallel_region
from TDATR.models.modules.mappings import gather_from_tensor_model_parallel_region
from TDATR.models.modules.mappings import reduce_from_tensor_model_parallel_region
from TDATR.models.modules.mappings import scatter_to_tensor_model_parallel_region
from TDATR.models.modules.mappings import reduce_scatter_to_sequence_parallel_region


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        padding_idx: pad token index.
        weight_init_method: method to initialize weights.
        bias_init_method: method to initialize bias.
        dtype: parameters dtype, default=torch.float16.
        use_cpu_initialization: If set, affine parallel parameters initialization uses CPU
        keep_master_weight_for_test: This was added for testing and should be set to False.
                                     It returns the master weights used for initialization.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int, 
                 init_method: Optional[Callable]=init.xavier_normal_,
                 dtype: Optional[torch.dtype]=torch.float16,
                 use_cpu_initialization: Optional[bool]=True,
                 keep_master_weight_for_test: Optional[bool]=False):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.use_cpu_initialization = use_cpu_initialization
        self.init_method = init_method
        self.keep_master_weight_for_test = keep_master_weight_for_test

        # Set the detauls for compatibility.
        self.padding_idx = padding_idx
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = gpc.get_world_size(ParallelMode.TENSOR)
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, gpc.get_local_rank(ParallelMode.TENSOR),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        if self.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=self.dtype))
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=self.dtype))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.use_cpu_initialization:
            initialize_weight_cpu(
                self.weight, 
                partition_dim=0, 
                per_partition_size=self.num_embeddings_per_partition,
                init_method=self.init_method, 
            )
        else:
            initialize_weight_gpu(self.weight, self.init_method, partition_dim=0)

    def forward(self, input_):
        input_mask = None
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            assert input_mask is not None
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output

class LinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication and gradient accumulation
    fusion in backprop.
    """

    @staticmethod
    def forward(
        ctx, 
        input, 
        weight, 
        bias,
        async_grad_allreduce, 
        sequence_parallel,
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias.requires_grad if bias is not None else False
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
      
        if sequence_parallel:
            world_size = gpc.get_world_size(ParallelMode.TENSOR)
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size
            all_gather_buffer = global_memory_buffer.get_tensor(
                tensor_shape=dim_size,
                dtype=input.dtype,
                name="mpu"
            )
            if hasattr(torch.distributed, "all_gather_into_tensor"):
                torch.distributed.all_gather_into_tensor(
                    all_gather_buffer,
                    input,
                    group=gpc.get_group(ParallelMode.TENSOR)
                )
            else:
                torch.distributed._all_gather_base(
                    all_gather_buffer,
                    input,
                    group=gpc.get_group(ParallelMode.TENSOR)
                )
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())

        if bias is not None:
            output = output + bias
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        
        handle = None
        if ctx.sequence_parallel:
            world_size = gpc.get_world_size(ParallelMode.TENSOR)
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size
            all_gather_buffer = global_memory_buffer.get_tensor(
                tensor_shape=dim_size,
                dtype=input.dtype,
                name="mpu"
            )
            if hasattr(torch.distributed, "all_gather_into_tensor"):
                handle = torch.distributed.all_gather_into_tensor(
                    all_gather_buffer,
                    input,
                    group=gpc.get_group(ParallelMode.TENSOR), 
                    async_op=True
                )
            else:
                handle = torch.distributed._all_gather_base(
                    all_gather_buffer,
                    input,
                    group=gpc.get_group(ParallelMode.TENSOR), 
                    async_op=True
                )
            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
            total_input = all_gather_buffer
        else:
            total_input = input

        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        total_input = total_input.reshape(-1, total_input.shape[-1])
 
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, 
                group=gpc.get_group(ParallelMode.TENSOR), 
                async_op=True
            )
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        sub_grad_input = None
        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            if hasattr(torch.distributed, "reduce_scatter_tensor"):
                handle = torch.distributed.reduce_scatter_tensor(
                    sub_grad_input, 
                    grad_input, 
                    group=gpc.get_group(ParallelMode.TENSOR),
                    async_op=True
                )
            else:
                handle = torch.distributed._reduce_scatter_base(
                    sub_grad_input, 
                    grad_input, 
                    group=gpc.get_group(ParallelMode.TENSOR),
                    async_op=True
                )
            # Delay the start of weight gradient computation shortly (3us) to have
            # reduce scatter scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
        
        grad_weight = grad_output.t().matmul(total_input) if weight.requires_grad else None
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        partition_stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: Optional[bool]=True,
                 weight_init_method: Optional[Callable]=init.xavier_normal_,
                 bias_init_method: Optional[Callable]=init.zeros_,
                 dtype: Optional[torch.dtype]=torch.float16,
                 partition_stride: Optional[int]=1,
                 skip_bias_add: Optional[bool]=False,
                 gather_output: Optional[bool]=True,
                 use_cpu_initialization: Optional[bool]=True,
                 keep_master_weight_for_test: Optional[bool]=False,
                 only_return_output: Optional[bool]=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.weight_init_method = weight_init_method
        self.bias_init_method = bias_init_method
        self.dtype = dtype
        self.partition_stride = partition_stride
        self.skip_bias_add = skip_bias_add
        self.gather_output = gather_output
        self.use_cpu_initialization = use_cpu_initialization
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.only_return_output = only_return_output

        # Divide the weight matrix along the last dimension.
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.output_size_per_partition = divide(output_size, world_size)

        self.async_tensor_model_parallel_allreduce = (
                gpc.config.model_parallel.async_tensor_model_parallel_allreduce and
                world_size > 1)
        self.sequence_parallel = (
                gpc.config.model_parallel.sequence_parallel and
                world_size > 1)
        assert not self.async_tensor_model_parallel_allreduce or \
            not self.sequence_parallel, "async_tensor_model_parallel_allreduce \
                and sequence_parallel cannot be true at the same time"
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.

        if self.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=self.dtype))
        else:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                device=torch.cuda.current_device(),
                                                dtype=self.dtype))

        if bias:
            if self.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size_per_partition, 
                                                  dtype=self.dtype))
            else:
                self.bias = Parameter(torch.empty(self.output_size_per_partition,
                                                  device=torch.cuda.current_device(),
                                                  dtype=self.dtype))
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
                per_partition_size=self.output_size_per_partition,
                init_method=self.weight_init_method,
                partition_stride=self.partition_stride,
            )
            if self.bias is not None:
                initialize_weight_cpu(
                    self.bias, 
                    partition_dim=0, 
                    per_partition_size=self.output_size_per_partition,
                    init_method=self.bias_init_method,
                    partition_stride=self.partition_stride
                )
        else:
            initialize_weight_gpu(
                self.weight,
                self.weight_init_method,
                partition_dim=0,
                partition_stride=self.partition_stride,
            )
            if self.bias is not None:
                initialize_weight_gpu(
                    self.bias,
                    self.bias_init_method,
                    partition_dim=0,
                    partition_stride=self.partition_stride
                )
  
    def forward(self, input_):
        # input_ = input_.clamp(-8, 8)
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        output_parallel = LinearWithAsyncCommunication.apply(
            input_parallel,                              # input
            self.weight,                                 # weight
            bias,                                        # bias
            self.async_tensor_model_parallel_allreduce,  # async_grad_allreduce
            self.sequence_parallel,                      # sequence_parallel
        )

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        if self.only_return_output and not self.skip_bias_add:
            return output
        
        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        partition_stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: Optional[bool]=True,
                 weight_init_method: Optional[Callable]=init.xavier_normal_,
                 bias_init_method: Optional[Callable]=init.zeros_,
                 dtype: Optional[torch.dtype]=torch.float16,
                 partition_stride: Optional[int]=1,
                 skip_bias_add: Optional[bool]=False,
                 input_is_parallel: Optional[bool]=False,
                 use_cpu_initialization: Optional[bool]=True,
                 keep_master_weight_for_test: Optional[bool]=False,
                 only_return_output: Optional[bool]=False):
        super(RowParallelLinear, self).__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.weight_init_method = weight_init_method
        self.bias_init_method = bias_init_method
        self.dtype = dtype
        self.partition_stride = partition_stride
        self.skip_bias_add = skip_bias_add
        self.input_is_parallel = input_is_parallel
        self.use_cpu_initialization = use_cpu_initialization
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.only_return_output = only_return_output
        # Divide the weight matrix along the last dimension.
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.input_size_per_partition = divide(input_size, world_size)
        
        self.sequence_parallel = gpc.config.model_parallel.sequence_parallel
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.

        if self.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=self.dtype))
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=self.dtype))
        if bias:
            if self.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=self.dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=self.dtype))
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if inspect.signature(self.bias_init_method).parameters.get("weight") is not None:
            self.bias_init_method = partial(self.bias_init_method, weight=self.weight)
        if self.use_cpu_initialization:
            initialize_weight_cpu(
                self.weight, 
                partition_dim=1, 
                per_partition_size=self.input_size_per_partition,
                init_method=self.weight_init_method,
                partition_stride=self.partition_stride,
            )
            if self.bias is not None:
                initialize_weight_cpu(
                    self.bias,
                    partition_dim=None,
                    init_method=self.bias_init_method, 
                )
        else:
            initialize_weight_gpu(
                self.weight,
                self.weight_init_method,
                partition_dim=1,
                partition_stride=self.partition_stride
            )
            if self.bias is not None:
                initialize_weight_gpu(
                    self.bias,
                    self.bias_init_method,
                    partition_dim=None
                )
    
    def forward(self, input_):
        # input_ = input_.clamp(-8, 8)
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        output_parallel = LinearWithAsyncCommunication.apply(
            input_parallel,      # input
            self.weight,         # weight
            None,                # bias
            None,                # async_grad_allreduce
            None,                # sequence_parallel
        )

        # All-reduce across all the partitions.
        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias

        if self.only_return_output and not self.skip_bias_add:
            return output

        return output, output_bias
