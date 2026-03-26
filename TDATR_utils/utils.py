import logging
import math
import os
import sys
import importlib
import torch
import torch.nn.init as init
from omegaconf import DictConfig, OmegaConf, open_dict, _utils
from dataclasses import _MISSING_TYPE, MISSING, is_dataclass

from TDATR_utils.call_main import infer_init_method
from TDATR_utils.global_context import global_context as gpc
from typing import Dict, List, Tuple, Union, Optional,Callable
import torch.distributed as dist
from torch import Tensor



logger = logging.getLogger(__name__)


def split_tensor(tensor: torch.Tensor,
                 num_partitions: int,
                 dim: int,
                 contiguous_split_chunks: Optional[bool]=False) -> List[torch.Tensor]:
    """Split a tensor along given dim.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        dim: split dim
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    dim_size = divide(tensor.size()[dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, dim_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""

    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Filteration based on the cumulative sum.
    filter_ = cumulative_probs > top_p
    # This shift by 1 is weird and I cannot justify it. This existed
    # in the original implementation:
    #   https://github.com/ari-holtzman/degen/blob/master/gen.py
    # and I guess it is needed so keeping it for now.
    filter_[:, 1:] = filter_[:, :-1].clone()
    # Make sure we at least have one token to select from.
    filter_[..., 0] = 0

    # Fill in the filtered part
    filter_ = filter_.scatter(1, sorted_indices, filter_)
    logits.masked_fill_(filter_, float('-Inf'))



def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf."""

    filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(filter_, float('-Inf'))

def sample(logits, top_k=0, top_p=0.0, temperature=1.0, vocab_size=None):
    """ Sample and generate a token.
    Note: logits has the dimension [b, v] where b is the batch size
          and v is the vocabulary size.
    If vocab_size is provided, we will make sure the sample that is
    generated is in [0, vocab-size). This will avoid out of vocabulary
    generations due to padding.
    """

    # Check logits for consistency.
    assert logits.ndim == 2, 'expected the logits to be of [b, v] shape.'
    # assert logits.type() == 'torch.cuda.FloatTensor', \
    #     'input logits should be floats.'


    # Greedy is just simple argmax.
    if top_k == 1:
        assert top_p == 0.0, 'cannot set both greedy and top-p samplings.'
        samples = torch.argmax(logits, dim=-1)

    # Top-k or top-p sampling.
    else:
        # Clone so we do not modify the inputs,
        logits = logits.clone()
        # Apply temperature in place.
        if temperature != 1.0:
            logits.div_(temperature)

        if top_k > 1:
            assert top_p == 0.0, 'cannot set both top-k and top-p samplings.'
            assert top_k <= logits.size(1), 'top-k is larger than logit size.'
            if vocab_size:
                assert top_k < vocab_size, 'top-k is larger than vocab size.'
            modify_logits_for_top_k_filtering(logits, top_k)

        elif top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
            modify_logits_for_top_p_filtering(logits, top_p)

        # After filtering, we need to recalculate the distribution.
        probs = logits.softmax(dim=-1)
        samples = torch.multinomial(probs.float(), num_samples=1).view(-1)

    # If vocab size is provided, make sure the samples are in
    # in the range [0, vocab-size).
    if vocab_size:
        samples = torch.clamp(samples, min=0, max=(vocab_size - 1))

    return samples

def broadcast_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Broadcast tensor values from last stage into the first stage."""

    is_last_stage = gpc.is_pipeline_last_stage()
    is_first_stage = gpc.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return tensor
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        if is_last_stage:
            _is_cuda_contiguous(tensor)
        else:
            tensor = torch.empty(size,
                                 dtype=dtype,
                                 device=torch.cuda.current_device())
        src = gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1]
        group = gpc.get_group(ParallelMode.PIPELINE)
        # Broadcast from last stage into the first stage.
        dist.broadcast(tensor, src, group)
    else:
        tensor = None

    return tensor

def broadcast_from_last_pipeline_stage(size, dtype, tensor=None):
    """Broadcast a tensor from last pipeline stage to all ranks."""

    is_last_stage = gpc.is_pipeline_last_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if gpc.is_pipeline_first_stage() and is_last_stage:
        return tensor

    if is_last_stage:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())
    # Get the group and corresponding source rank.
    src = gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1]
    group = gpc.get_group(ParallelMode.PIPELINE)
    dist.broadcast(tensor, src, group)

    return tensor

def copy_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Copy tensor values from last stage into the first stage.
    Note that the input tensor is updated in place."""

    is_last_stage = gpc.is_pipeline_last_stage()
    is_first_stage = gpc.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        _is_cuda(tensor)
        is_contiguous = tensor.is_contiguous()
        src = gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1]
        group = gpc.get_group(ParallelMode.PIPELINE)
        if is_contiguous:
            tensor_ = tensor
        else:
            if is_last_stage:
                tensor_ = tensor.contiguous()
            else:
                tensor_ = torch.empty(size,
                                      dtype=dtype,
                                      device=torch.cuda.current_device())
        # Broadcast from last stage into the first stage.
        dist.broadcast(tensor_, src, group)
        # Update the first stage tensor
        if is_first_stage and not is_contiguous:
            tensor[...] = tensor_

def broadcast_float_list(size, float_list=None, rank=0):
    """Broadcast a list of float values."""

    return broadcast_list(size, torch.float32, list_values=float_list, rank=rank)


def broadcast_list(size, dtype, list_values=None, rank=0):
    """Broadcast a list of values with a given type."""
    world_size = gpc.get_world_size(ParallelMode.MODEL)
    if world_size == 1:
        return torch.tensor(list_values, dtype=dtype,
                            device=torch.cuda.current_device())
    tensor = None
    if gpc.get_local_rank(ParallelMode.MODEL) == rank:
        tensor = torch.tensor(list_values, dtype=dtype,
                              device=torch.cuda.current_device())

    return broadcast_tensor(size, dtype, tensor=tensor, rank=rank)


def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda


def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()

def broadcast_tensor(size, dtype, tensor=None, rank=0):
    """ Given size and type of a tensor on all ranks and the tensor value
        only on a specific rank, broadcast from that rank to all other ranks.
    """
    group = gpc.get_group(ParallelMode.MODEL)
    world_size = gpc.get_world_size(ParallelMode.MODEL)
    if world_size == 1:
        return tensor

    if gpc.get_local_rank(ParallelMode.MODEL) == rank:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())

    src = gpc.get_ranks_in_group(ParallelMode.MODEL)[rank]
    dist.broadcast(tensor, src, group=group)

    return tensor

class VocabUtility(object):
    """Split the vocabulary into `world_size` chunks amd return the
        first and last index of the vocabulary belonging to the `rank`
        partition: Note that indecies in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int,
                                                  rank: int,
                                                  world_size: int) -> Tuple[int, int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int,
                                           rank: int,
                                           world_size: int) -> Tuple[int, int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size)

def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, \
        f'{numerator} is not divisible by {denominator}'


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def nn_conv_weight_init_(weight: torch.Tensor):
    init.kaiming_uniform_(weight, a=math.sqrt(5))


def nn_conv_bias_init_(bias: torch.Tensor,
                       weight: torch.Tensor) -> None:
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

def import_func(tasks_dir, namespace):
    for file in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            task_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + task_name)

def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        

        # ensure that user modules are only imported once
        import_user_module.memo = getattr(import_user_module, "memo", set())
        if module_path not in import_user_module.memo:
            import_user_module.memo.add(module_path)

            module_parent, module_name = os.path.split(module_path)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)

                models_path = os.path.join(module_path, "models")
                if os.path.exists(models_path):

                    import_func(models_path, f"{module_name}.models")

                tokenizers_path = os.path.join(module_path, "tokenizers")
                if os.path.exists(tokenizers_path):

                    import_func(tokenizers_path, f"{module_name}.tokenizers")
            else:
                raise ImportError(
                    "Failed to import --user-dir={} because the corresponding module name "
                    "({}) is not globally unique. Please rename the directory to "
                    "something unique and try again.".format(module_path, module_name)
                )

def merge_with_parent(dc , cfg: DictConfig, remove_missing=True):
    if remove_missing:

        if is_dataclass(dc):
            target_keys = set(dc.__dataclass_fields__.keys())
        else:
            target_keys = set(dc.keys())

        with open_dict(cfg):
            for k in list(cfg.keys()):
                if k not in target_keys:
                    del cfg[k]

    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg

def add_defaults(cfg: DictConfig) -> None:
    """This function adds default values that are stored in dataclasses that hydra doesn't know about """

    from typing import Any

    OmegaConf.set_struct(cfg, False)

    for k, v in {"model":None, "tokenizer":None}.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None:
            dc = None

            if isinstance(field_cfg, str):
                field_cfg = DictConfig({"_name": field_cfg})
                field_cfg.__dict__["_parent"] = field_cfg.__dict__["_parent"]

            name = getattr(field_cfg, "_name", None)
            if k=="model":
                from TDATR.models.mini_gpt4_ipt_v2 import MiniGPT4Config

                dc = MiniGPT4Config

                if dc is not None:
                    cfg[k] = merge_with_parent(dc, field_cfg)
            else:
                from TDATR.tokenizers.bbox_tokenizer import BboxTokenConfig

                dc = BboxTokenConfig

                if dc is not None:
                    cfg[k] = merge_with_parent(dc, field_cfg)

    assert cfg.model is not None, 'Missing model config!'

from enum import Enum


from TDATR_utils.global_variables import ParallelMode
def distributed_main(i, main, cfg, kwargs):
    cfg.distributed_training.device_id = i
    #if torch.cuda.is_available() and not cfg.common.cpu:
    #    torch.cuda.set_device(cfg.distributed_training.device_id)
    if cfg.distributed_training.distributed_rank is None:  # torch.multiprocessing.spawn
        cfg.distributed_training.distributed_rank = kwargs.pop("start_rank", 0) + i

    after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
    if after_distributed_init_fn:
        cfg = after_distributed_init_fn(cfg)

    main(cfg, **kwargs)

    if torch.distributed.is_initialized():
        torch.distributed.barrier(gpc.get_group(ParallelMode.GLOBAL))


def call_main(cfg, main, **kwargs):
    if cfg.distributed_training.distributed_init_method is None:
        infer_init_method(cfg.distributed_training)
    if cfg.distributed_training.distributed_init_method is not None:
        # distributed training
        if not cfg.distributed_training.distributed_no_spawn:
            start_rank = cfg.distributed_training.distributed_rank
            cfg.distributed_training.distributed_rank = None  # assign automatically
            kwargs["start_rank"] = start_rank 
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(main, cfg, kwargs),
                nprocs=min(
                    torch.cuda.device_count(),
                    cfg.distributed_training.distributed_world_size,
                ),
                join=True,
            )
        else:
            distributed_main(cfg.distributed_training.device_id, main, cfg, kwargs)
    else:
        # single GPU main
        main(cfg, **kwargs)

try:
    from hydra import compose, initialize
except ImportError:
    from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from omegaconf import DictConfig, OmegaConf, open_dict, _utils
from argparse import ArgumentError, ArgumentParser, Namespace

class omegaconf_no_object_check:
    def __init__(self):
        self.old_is_primitive = _utils.is_primitive_type

    def __enter__(self):
        _utils.is_primitive_type = lambda _: True

    def __exit__(self, type, value, traceback):
        _utils.is_primitive_type = self.old_is_primitive


def _set_legacy_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, "add_args"):
        return

    import argparse

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, allow_abbrev=False
    )
    cls.add_args(parser)
    # copied from argparse.py:
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)

def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []
    return overrides, deletes

def convert_namespace_to_omegaconf(args: Namespace) -> DictConfig:
    """Convert a flat argparse.Namespace to a structured DictConfig."""

    # Here we are using field values provided in args to override counterparts inside config object
    overrides, deletes = override_module_args(args)

    # configs will be in hulk/config after installation
    config_path = os.path.join("..", "config")

    GlobalHydra.instance().clear()

    with initialize(config_path=config_path):
        try:
            composed_cfg = compose("config", overrides=overrides, strict=False)
        except:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise

        for k in deletes:
            composed_cfg[k] = None

    cfg = OmegaConf.create(
        OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils
    OmegaConf.set_struct(cfg, True)
    return cfg


def gen_parser_from_dataclass(
    parser: ArgumentParser,
    dataclass_instance,
    delete_default: bool = False,
    with_prefix: Optional[str] = None,
) -> None:
    """
        convert a dataclass instance to tailing parser arguments.

        If `with_prefix` is provided, prefix all the keys in the resulting parser with it. It means that we are
        building a flat namespace from a structured dataclass (see transformer_config.py for example).
    """

    def argparse_name(name: str):
        if name == "data" and (with_prefix is None or with_prefix == ''):
            # normally data is positional args, so we don't add the -- nor the prefix
            return name
        if name == "_name":
            # private member, skip
            return None
        full_name = "--" + name.replace("_", "-")
        if with_prefix is not None and with_prefix != '':
            # if a prefix is specified, construct the prefixed arg name
            full_name = with_prefix + "-" + full_name[2:]  # strip -- when composing
        return full_name



def _is_tensor_parallel(partition_dim: int) -> bool:
    if partition_dim is None:
        return False
    if gpc.get_world_size(ParallelMode.TENSOR) <= 1:
        return False
    return True


from contextlib import contextmanager, nullcontext

@contextmanager
def init_tensor_parallel_parameters(is_gpu_initializer=True):
    torch_calc_fan_in_fan_out = torch.nn.init._calculate_fan_in_and_fan_out
    def _calculate_fan_in_and_fan_out(tensor):
        fan_in, fan_out = torch_calc_fan_in_fan_out(tensor)
        if not getattr(tensor, _TENSOR_PARALLEL, False):
            return fan_in, fan_out # 1, 0
        tensor_dim = tensor.dim()
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
        partition_dim = getattr(tensor, _PARTITION_DIM)
        if not isinstance(partition_dim, int) or not (0 <= partition_dim < tensor_dim):
            raise ValueError(f"Detected partition_dim={partition_dim}, but tensor dim={tensor_dim}.")
        if partition_dim == 0:
            fan_out *= world_size
        elif partition_dim == 1:
            fan_in  *= world_size
        elif tensor_dim > 2 and partition_dim >= 2:
            fan_out *= world_size
            fan_in  *= world_size
        return fan_in, fan_out

    torch.nn.init._calculate_fan_in_and_fan_out = _calculate_fan_in_and_fan_out
    context = nullcontext()
    with context:
        yield
    torch.nn.init._calculate_fan_in_and_fan_out = torch_calc_fan_in_fan_out



def broadcast(tensor: Tensor, src: int, parallel_mode: ParallelMode, async_op: bool = False):
    r"""Broadcast tensors to whole parallel group. Tensor must have the same
    number of elements in all processes participating in the collective.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be broadcast.
        src (int): Source rank.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The tensor need to be broadcast only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        if out.data.dtype == torch.bfloat16:
            assert not async_op, "BF16 does not support async broadcast"
            if gpc.get_world_size(parallel_mode) != 1:
                out = out.to(torch.cuda.current_device())
                dist.broadcast(
                    out,
                    src=src,
                    group=gpc.get_group(parallel_mode)
                )
                out = out.cpu()
        else:
            group = gpc.get_cpu_group(parallel_mode) if tensor.device.type == "cpu" else gpc.get_group(parallel_mode)
            work = dist.broadcast(out, src=src, group=group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out

_TENSOR_PARALLEL = "tensor_model_parallel"
_PARTITION_DIM = "partition_dim"
_PARTITION_STRIDE = "partition_stride"


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {_TENSOR_PARALLEL: False,
                                      _PARTITION_DIM: -1,
                                      _PARTITION_STRIDE: 1}


def set_tensor_model_parallel_attributes(tensor: torch.Tensor,
                                         is_parallel: int,
                                         dim: int,
                                         partition_stride: int):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, _TENSOR_PARALLEL, is_parallel)
    setattr(tensor, _PARTITION_DIM, dim)
    setattr(tensor, _PARTITION_STRIDE, partition_stride)

@torch.no_grad()
def initialize_weight_gpu(weight: torch.Tensor,
                          init_method: Callable,
                          partition_dim: Optional[int]=None,
                          partition_stride:Optional[int]=1):
    assert weight.device.type == 'cuda' or weight.device.type == 'npu', f"Expected a cuda tensor, but got {weight.device}."

    is_parallel = _is_tensor_parallel(partition_dim)
    if not is_parallel:
        partition_dim = None
    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=is_parallel,
                                         dim=partition_dim,
                                         partition_stride=partition_stride)
    with init_tensor_parallel_parameters():
        init_method(weight)
    if not is_parallel: # is a global tensor
        broadcast(weight, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], ParallelMode.TENSOR)

def get_1d_parallel_split_size(dim_size:int,
                               partition_stride: int=1,
                               world_size: Optional[int]=None) -> Tuple[int, int]:
    """If we split tensor by world size and stride along a dim,
    this function will return the number of chunks and size of each chunk."""
    if world_size is None:
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
    per_partition_size = divide(dim_size, world_size)
    per_partition_per_stride_size = divide(per_partition_size, partition_stride)
    num_chunks = partition_stride * world_size
    return num_chunks, per_partition_per_stride_size


def get_global_tensor_size(dim_size:int, world_size: Optional[int]=None) -> int:
    """Compute the global tensor dim size"""
    if world_size is None:
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
    return dim_size * world_size
def get_1d_parallel_split_indices(rank:Optional[int]=None, 
                                  partition_stride: int=1,
                                  world_size: Optional[int]=None) -> Tuple[int]:
    """If we split tensor by world size and stride along a dim,
    this function will return the indices of splited chunks on
    the current member in TP group."""
    if rank is None:
        rank = gpc.get_local_rank(ParallelMode.TENSOR)
    if world_size is None:
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
    num_chunks = partition_stride * world_size
    return tuple(range(rank, num_chunks, world_size))


def split_tensor_to_local(tensor: torch.Tensor,
                          partition_dim:int,
                          partition_stride: Optional[int],
                          out:Optional[torch.Tensor]=None,
                          world_size: Optional[int]=None,
                          cur_rank: Optional[int]=None) -> torch.Tensor:
    """Split a tensor by tensor parallel size and stride along a dim.

    If we want to split a global tensor along `dim` to each member of tensor parallel group,
    we first split weight to `world_size*stride` partitions, each partition size equal to 
    `dim_size//(world_size*stride)`.

    with world_size=4, stride=1, each member of tensor parallel groups gets the indices of partitions:
        rank=0: [0]
        rank=1: [1]
        rank=2: [2]
        rank=3: [3]
    with world_size=4, stride=2, each member of tensor parallel groups gets the indices of partitions:
        rank=0: [0, 4]
        rank=1: [1, 5]
        rank=2: [2, 6]
        rank=3: [3, 7]
    with world_size=4, stride=4, each member of tensor parallel groups gets the indices of partitions:
        parallel rank=0: [0, 4, 8, 12]
        parallel rank=1: [1, 5, 9, 13]
        parallel rank=2: [2, 6, 10, 14]
        parallel rank=3: [3, 7, 11, 15]
    """
    if world_size is None:
        world_size = gpc.get_world_size(ParallelMode.TENSOR)
    if cur_rank is None:
        cur_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    if world_size == 1:
        return tensor if out is None else out.data.copy_(tensor)
    num_chunks, per_partition_per_stride_size = \
        get_1d_parallel_split_size(dim_size=tensor.size(partition_dim),
                                   partition_stride=partition_stride,
                                   world_size=world_size)
    assert per_partition_per_stride_size * num_chunks == tensor.size(partition_dim)
    tensor_chunks = torch.chunk(tensor.data, num_chunks, dim=partition_dim)
    partition_indices = get_1d_parallel_split_indices(rank=cur_rank,
                                                      partition_stride=partition_stride,
                                                      world_size=world_size)
    if out is not None and isinstance(out, torch.Tensor):
        return torch.cat([tensor_chunks[i] for i in partition_indices], dim=partition_dim, out=out)
    tensor_partition = torch.cat([tensor_chunks[i] for i in partition_indices], dim=partition_dim)
    return tensor_partition.contiguous()

@torch.no_grad()
def initialize_weight_cpu(weight: torch.Tensor,
                          partition_dim: Optional[int]=None,
                          per_partition_size: Optional[int]=0,
                          init_method: Optional[Callable]=None,
                          partition_stride: Optional[int]=1) -> Union[None, torch.Tensor]:
    assert weight.device.type == 'cpu', f"Expected a cpu tensor, but got {weight.device}."
    is_parallel = _is_tensor_parallel(partition_dim)
    if not is_parallel:
        partition_dim = None
    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=is_parallel,
                                         dim=partition_dim,
                                         partition_stride=partition_stride)

    shape = list(weight.shape)
    if is_parallel:
        tp_world_size = gpc.get_world_size(ParallelMode.TENSOR)
        shape[partition_dim] = shape[partition_dim] * tp_world_size
    master_weight = torch.empty(shape, dtype=torch.float32, requires_grad=False)
    with init_tensor_parallel_parameters(is_gpu_initializer=False):
        init_method(master_weight)
    master_weight = master_weight.to(dtype=weight.dtype)

    if is_parallel:
        assert per_partition_size != 0, "per_partition_size can't be 0 if partition_dim is not None"
        # Split and copy
        split_tensor_to_local(master_weight, partition_dim, partition_stride, weight)
    else: # global tensor
        weight.data = master_weight.data.clone()
        broadcast(weight, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], ParallelMode.TENSOR)
