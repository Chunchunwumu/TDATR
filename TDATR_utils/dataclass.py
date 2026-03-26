 # Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

import torch

from TDATR_utils.constants import (
    DATASET_IMPL_CHOICES,
    DDP_BACKEND_CHOICES,
    DDP_COMM_HOOK_CHOICES,
    GENERATION_CONSTRAINTS_CHOICES,
    GENERATION_DECODING_FORMAT_CHOICES,
    LOG_FORMAT_CHOICES,
    PRINT_ALIGNMENT_CHOICES,
    ZERO_SHARDING_CHOICES,
    RECOMPUTE_GRANULARITY_CHOICES,
    CLIP_GRAD_NORM_TYPE_CHOICES,
    TENSOR_PARALLEL_MODES,
    TENSOR_SHARD_STRATEGY,
    TENSOR_PLACEMENT_POLICY,
    CKPT_IO_STRATEGY,
    SEQ_PARALLEL_ALGO,
    ACT_QUANT_STRATEGY,
    WEIGHT_QUANT_STRATEGY

)

from omegaconf import II, MISSING


@dataclass
class HulkDataclass:
    """base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")

    @classmethod
    def from_namespace(cls, args):
        if isinstance(args, cls):
            return args
        else:
            config = cls()
            for k in config.__dataclass_fields__.keys():
                if k.startswith("_"):
                    # private member, skip
                    continue
                if hasattr(args, k):
                    setattr(config, k, getattr(args, k))

            return config


@dataclass
class CommonConfig(HulkDataclass):
    # This is the core dataclass including common parameters shared by all different jobs. Please append your params to other dataclasses if they were
    # used for a particular purpose or task, such as those dedicated for `distributed training`, `optimization`, etc.
    no_progress_bar: bool = field(
        default=False, metadata={"help": "disable progress bar"}
    )
    log_interval: int = field(
        default=100,
        metadata={
            "help": "log progress every N batches (when progress bar is disabled)"
        },
    )
    log_format: Optional[LOG_FORMAT_CHOICES] = field(
        default=None, metadata={"help": "log format to use"}
    )
    disable_dist_logging: bool = field(
        default=False, metadata={"help": "disable output log to file on all rank."}
    )
    tensorboard_logdir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to save logs for tensorboard, should match --logdir "
            "of running tensorboard (default: no tensorboard logging)"
        },
    )
    tensorboard_tag_prefix: str = field(
        default='base',
        metadata={
            "help": "tag prefix for tensorboard, the tags in tensorboard will"
            "look like this: base/loss, base/best_loss ... (default: base)"
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights and Biases project name to use for logging"},
    )
    azureml_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Log scalars to AzureML context"},
    )
    seed: int = field(
        default=1, metadata={"help": "pseudo random number generator seed"}
    )
    debug: bool = field(
        default=False, metadata={"help": "print debug info if set to true"}
    )
    layernorm_fp32: bool = field(
        default=False, metadata={"help": "print debug info if set to true"}
    )
    fp32_residual_connection: bool = field(
        default=False, metadata={"help": "If true, move residual connections to fp32."}
    )
    apply_layernorm_1p: bool = field(
        default=False,
        metadata={
            "help": "Adjust LayerNorm weights such that they are centered " \
                    "around zero. This improves numerical stability."
        }
    )
    cpu: bool = field(default=False, metadata={"help": "use CPU instead of CUDA"})
    npu: bool = field(default=False, metadata={"help": "use NPU instead of CUDA"})
    npu_profile_enabled: bool = field(default=False, metadata={'hele': "profile npu execute informations."})
    npu_jit_compile: bool = field(default=True, metadata={"help": "set npu compile mode to jit compile, should be False in dynamic shape scene."})
    fp16: bool = field(default=False, metadata={"help": "use FP16"})
    bf16: bool = field(default=False, metadata={"help": "use BF16"})
    memory_efficient_fp16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of FP16 training; implies --fp16"
        },
    )
    memory_efficient_bf16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of BF16 training; implies --bf16"
        },
    )
    fp16_no_flatten_grads: bool = field(
        default=False, metadata={"help": "don't flatten FP16 grads tensor"}
    )
    fp16_init_scale: int = field(
        default=2 ** 7, metadata={"help": "default FP16 loss scale"}
    )
    fp16_scale_window: Optional[int] = field(
        default=None,
        metadata={"help": "number of updates before increasing loss scale"},
    )
    fp16_scale_tolerance: float = field(
        default=0.0,
        metadata={
            "help": "pct of updates that can overflow before decreasing the loss scale"
        },
    )
    on_cpu_convert_precision: bool = field(
        default=False,
        metadata={
            "help": "if set, the floating point conversion to fp16 runs on CPU. "
            "This reduces bus transfer time and GPU memory usage."
        },
    )
    min_loss_scale: float = field(
        default=1e-4,
        metadata={
            "help": "minimum FP16/AMP loss scale, after which training is stopped"
        },
    )
    threshold_loss_scale: Optional[float] = field(
        default=None, metadata={"help": "threshold FP16 loss scale from below"}
    )
    amp: bool = field(default=False, metadata={"help": "use automatic mixed precision"})
    amp_batch_retries: int = field(
        default=2,
        metadata={
            "help": "number of retries of same batch after reducing loss scale with AMP"
        },
    )
    amp_init_scale: int = field(
        default=2 ** 7, metadata={"help": "default AMP loss scale"}
    )
    amp_scale_window: Optional[int] = field(
        default=None,
        metadata={"help": "number of updates before increasing AMP loss scale"},
    )
    user_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to a python module containing custom extensions (tasks and/or architectures)"
        },
    )
    empty_cache_freq: int = field(
        default=0,
        metadata={"help": "how often to clear the PyTorch CUDA cache (0 to disable)"},
    )
    all_gather_list_size: int = field(
        default=16384,
        metadata={"help": "number of bytes reserved for gathering stats from workers"},
    )
    quantization_config_path: Optional[str] = field(
        default=None, metadata={"help": "path to quantization config file"}
    )
    profile: bool = field(
        default=False, metadata={"help": "enable autograd profiler emit_nvtx"}
    )
    suppress_crashes: bool = field(
        default=False,
        metadata={
            "help": "suppress crashes when training with the hydra_train entry point so that the "
            "main method can return a value (useful for sweeps)"
        },
    )
    log_nvidia_smi: bool = field(
        default=False, 
        metadata={
            "help": "log output from nvidia-smi during training"
        },
    )
    log_time_throughput: bool = field(
        default=False,
        metadata={
            "help": "log distributed op elapsed time and throughtput(Bytes) during training"
        },
    )
    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "if True, causes cuDNN to benchmark algorithms and select the fastest."
        }
    )
    cudnn_deterministic: bool = field(
        default=False,
        metadata={
            "help": "if True, causes cuDNN to only use deterministic algorithms"
        }
    )
    cudnn_enabled: bool = field(
        default=True,
        metadata={
            "help": "if True, causes cuDNN to benchmark algorithms and select the fastest."
        }
    )
    dump_enabled: bool = field(
        default=False,
        metadata={
            "help": "if True, will dump params/activations/grads at every steps."
        }
    )
    experiment_dir: str = field(
        default='./',
        metadata={
            "help": "set experiment save dir, the checkpoint/tb/log files will be saved to this directory"
        }
    )
    log_file: str = field(
        default="train.log",
        metadata={
            "help": "log file name, default='train.log'"
        }
    )
    ddp_comm_monitor_enable: bool = field(
        default=False,
        metadata={
            "help": "if set, will add `dist.monitored_barrier` after each collective in DDP for monitoring comm timeout."
        }
    )
    ddp_comm_monitor_timeout: int = field(
        default=600,
        metadata={
            "help": "The number of seconds the monitor waits for communication."
        }
    )
    ddp_comm_monitor_port: int = field(
        default=39999,
        metadata={
            "help": "monitor server port number."
        }
    )
    gloo_group_enabled: bool = field(
        default=False,
        metadata={
            "help": "enable gloo group for cpu tensor communications."
        }
    )
    release_grad_for_optimizer: bool = field(
        default=False,
        metadata={
            "help": "release fp32 optimizer's grads memory when training with fp16/bf16"
        },
    )


@dataclass
class DistributedTrainingConfig(HulkDataclass):
    distributed_world_size: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "total number of GPUs across all nodes (default: all visible GPUs)"
        },
    )
    distributed_num_procs: Optional[int] = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "total number of processes to fork (default: all visible GPUs)"
        },
    )
    distributed_rank: Optional[int] = field(
        default=0, metadata={"help": "rank of the current worker"}
    )

    distributed_local_rank: Optional[int] = field(
        default=None, metadata={"help": "local rank of the current worker"}
    )
    distributed_backend: str = field(
        default="nccl", metadata={"help": "distributed backend"}
    )
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "typically tcp://hostname:port that will be used to "
            "establish initial connetion"
        },
    )
    distributed_master_addr: str = field(
        default='localhost',
        metadata={
            "help": "master addr (not required if using --distributed-init-method)"
        },
    )
    distributed_master_port: int = field(
        default=-1,
        metadata={
            "help": "port number (not required if using --distributed-init-method)"
        },
    )
    device_id: int = field(
        default=0,
        metadata={
            "help": "which GPU to use (usually configured automatically)",
            "argparse_alias": "--local_rank",
        },
    )
    distributed_no_spawn: bool = field(
        default=False,
        metadata={
            "help": "do not spawn multiple processes even if multiple GPUs are visible"
        },
    )
    ddp_backend: DDP_BACKEND_CHOICES = field(
        default="pytorch_ddp", metadata={"help": "DistributedDataParallel backend"}
    )
    ddp_comm_hook: DDP_COMM_HOOK_CHOICES = field(
        default="none", metadata={"help": "communication hook"}
    )
    bucket_cap_mb: int = field(
        default=25, metadata={"help": "bucket size for reduction"}
    )
    fix_batches_to_gpus: bool = field(
        default=False,
        metadata={
            "help": "don't shuffle batches between GPUs; this reduces overall "
            "randomness and may affect precision but avoids the cost of re-reading the data"
        },
    )
    find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "disable unused parameter detection (not applicable to "
            "--ddp-backend=legacy_ddp)"
        },
    )
    accumulate_allreduce_grads_in_fp32: bool = field(
        default=False,
        metadata={
            "help": "accumulate_allreduce_grads_in_fp32"
        },
    )
    gradient_as_bucket_view: bool = field(
        default=False,
        metadata={
            "help": "when set to True, gradients will be views pointing to different offsets of allreduce communication buckets. This can reduce peak memory usage, where the saved memory size will be equal to the total gradients size. "
            "--gradient-as-bucket-view=gradient_as_bucket_view)"
        },
    )
    fast_stat_sync: bool = field(
        default=False,
        metadata={"help": "[deprecated] this is now defined per Criterion"},
    )
    heartbeat_timeout: int = field(
        default=-1,
        metadata={
            "help": "kill the job if no progress is made in N seconds; "
            "set to -1 to disable"
        },
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as "
            "batchnorm population statistics"
        },
    )
    slowmo_momentum: Optional[float] = field(
        default=None,
        metadata={
            "help": "SlowMo momentum term; by default use 0.0 for 16 GPUs, "
            "0.2 for 32 GPUs; 0.5 for 64 GPUs, 0.6 for > 64 GPUs"
        },
    )
    slowmo_base_algorithm: str = field(
        default="localsgd",
        metadata={
            "help": "Base algorithm. Either 'localsgd' or 'sgp'. Please refer "
            "to the documentation of 'slowmo_base_algorithm' parameter in "
            "https://fairscale.readthedocs.io/en/latest/api/experimental/nn/slowmo_ddp.html "
            "for more details"
        },
    )
    localsgd_frequency: int = field(
        default=3, metadata={"help": "Local SGD allreduce frequency"}
    )
    nprocs_per_node: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "number of GPUs in each node. An allreduce operation across GPUs in "
            "a node is very fast. Hence, we do allreduce across GPUs in a node, "
            "and gossip across different nodes"
        },
    )
    zero_sharding: ZERO_SHARDING_CHOICES = field(
        default="none", metadata={"help": "ZeRO sharding"}
    )
    fp16: bool = II("common.fp16")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    
    # configuration for --ddp-backend=fully_sharded # TODO remove it
    no_reshard_after_forward: bool = field(
        default=False,
        metadata={"help": "don't reshard parameters after forward pass"},
    )
    fp32_reduce_scatter: bool = field(
        default=False,
        metadata={"help": "reduce-scatter grads in FP32"},
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "offload FP32 params to CPU"}
    )
    use_sharded_state: Optional[bool] = field(
        default=False, metadata={"help": "load and save local state dict"}
    )
    fsdp_gradient_predivide_factor: Optional[float] = field(
        default=None,
        metadata={"help": "factor to predivide gradients before reducee scatter"},
    )


@dataclass
class ModelParallelConfig(HulkDataclass):
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Degree of data model parallelism."}
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Sequential parallelism corss TP group: split in the T dimension to save the GPU " \
                          "memory of LayerNorm and Dropout layers cross tensor parallel group."}
    )
    sequence_parallel_size: int = field(
        default=1,
        metadata={"help": "Sequence parallel size."}
    )
    sequence_parallel_algo: SEQ_PARALLEL_ALGO = field(
        default='local_atten',
        metadata={"help": "name of sequence parallel algorithm."}
    )
    async_tensor_model_parallel_allreduce: bool = field(
        default=False,
        metadata={"help": "Whether asynchronous computing is enabled during parallel linear execution " \
                          "to enable simultaneous communication and computing"}
    )
    recompute_granularity: Optional[RECOMPUTE_GRANULARITY_CHOICES] = field(
        default="none",
        metadata={"help": "There are two types of recalculation granularity: " \
                        "(1) full: The whole Transformer layer uses the checkpointing mechanism; " \
                        "(2) selective: Just using the checkpoint mechanism for CoreAttention, " \
                        " which will occupy more memory than full, but the speed will be improved a lot"}
    )
    tensor_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Degree of tensor model parallelism."}
    )
    tensor_model_parallel_mode: TENSOR_PARALLEL_MODES = field(
        default="1d",
        metadata={"help": "The mode of tensor model parallelism."}
    )
    tensor_model_parallel_depth: int = field(
        default=2,
        metadata={"help": "The depth of 2.5d parallel"}
    )
    pipeline_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Degree of pipeline model parallelism."}
    )
    virtual_pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of virtual pipeline model parallelism."}
    )
    manual_pipeline_partition: Optional[str] = field(
        default=None,
        metadata={
            "help": "Using for partition model when using pipeline parallel training. for example, " \
                    "if we set manual_pipeline_partition=[[(0, 2), (4, 6)], [(2, 4), (6, 8)]], that " \
                    "means if we have 8 layers, 2 stages, and 2 virtual stages, we want an assignment of " \
                    "layers to stages like (each tuple is a model chunk): " \
                    "Stage 0: (0, 1)  (4, 5)" \
                    "Stage 1: (2, 3)  (6, 7)" \
        }
    )
    use_cpu_initialization: bool = field(
        default=False,
        metadata={
            "help": 'If set, affine parallel weights initialization uses CPU'
        }
    )
    global_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": 'Training batch size. If set, it should be a '
                          'multiple of micro-batch-size times data-parallel-size. '
                          'If this value is None, then use micro-batch-size * '
                          'data-parallel-size as the global batch size. '
                          'This choice will result in 1 for number of micro-batches.'}
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": 'Batch size per model instance (local batch size). '
                          'Global batch size is local batch size times data '
                          'parallel size times number of micro batches.'}
    )
    num_micro_batch: Optional[int] = field(
        default=None,
        metadata={"help": 'Batch size per model instance (local batch size). '
                          'Global batch size is local batch size times data '
                          'parallel size times number of micro batches.'}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    distribute_checkpointed_activations: bool = field(
        default=False,
        metadata={
            "help": "distribute offloaded checkpoints to tensor parallel gpus. "
            "It adds extra within node all_reduce but reduces checkpointed activations significantly,"
            "so a good way to trade speed for gpu memory."
        },
    )
    scatter_gather_tensors_in_pipeline: bool = field(
        default=True,
        metadata={
            "help": "Use scatter/gather to optimize communication of tensors in pipeline.",
        }
    )
    p2p_fixed_activations: bool = field(
        default=False,
        metadata={
            "help": "whether activation's shape and dtype are immutable between pipeline layers."
            "if the shape and dtype of activations change dynamically, we should send/recv activation"
            "metas before each p2p communication of activitions, this will lead to decrease training efficiency"
        }
    )
    p2p_activation_shapes: Optional[str] = field(
        default=None,
        metadata={
            "help": "activations shapes between pipeline layers, format as: [[1, 1024, 512],[1, 1024, 512]]"
            "if p2p_activation_shapes=None or p2p_activation_dtypes=None, we will send/recv activation metas"
            "once time in each batch training."
        }
    )
    p2p_activation_dtypes: Optional[str] = field(
        default=None,
        metadata={
            "help": "activations dtypes between pipeline layers, format as: [torch.float16, torch.float16]"
            "if p2p_activation_shapes=None or p2p_activation_dtypes=None, we will send/recv activation metas"
            "once time in each batch training."
        }
    )
    zero_shard_size: int = field(
        default=-1,
        metadata={
            "help": "shard size of zero-v2, zero_shard_size=-1 means use the data parallel size."
        }
    )


@dataclass
class ZeroConfig(HulkDataclass):
    shard_strategy: TENSOR_SHARD_STRATEGY = field(
        default='tensor',
        metadata={"help": "A shard strategy to manage shard behavior."}
    )
    reduce_scatter_bucket_size_mb: int = field(
        default=25,
        metadata={"help": "Reduce-scatter bucket size in *MB*. Defaults to 25."}
    )
    fp32_reduce_scatter:bool = field(
        default=False,
        metadata={"help": "If set to `True`, gradients are forced to FP32 before reduce-scatter. Defaults to False."}
    )
    tensor_placement_policy: TENSOR_PLACEMENT_POLICY = field(
        default='cuda',
        metadata={
            "help": "Which device to place *held* tensors. It can be 'cpu', 'cuda' and 'auto'." \
                    "If it's 'cuda', they won't be offloaded, which means max CUDA memory will be used." \
                    "If it's 'auto', they are moving dynamically based on CPU and CUDA memory usage." \
                    "It will utilize heterogeneous memory space evenly and well." \
                    "Note that 'auto' policy can only work well when no other processes use CUDA during your training." \
                    "Defaults to 'cuda'."    
            }
    )
    gradient_predivide_factor: float = field(
        default=1.0,
        metadata={"help": "Gradient is divived by this value before reduce-scatter. Defaults to 1.0."}
    )
    reuse_fp16_shard: bool = field(
        default=False,
        metadata={"help": "Whether to reuse fp16 shard for param and grad. Enabling this can reduce GPU memory usage, " \
                          "but you have to make sure you disable it when using gradient accumulation. In this mode, grad " \
                          "will be fp16. Make sure your optimizer supports mixed precision (fp32 param and fp16 grad)." \
                          "We find that PyTorch's optimizers don't support mixed precision, so we recommend you enable " \
                          "this only when using our CPUAdam with CPU offload. Defaults to False."}
    )
    gpu_margin_mem_ratio: float = field(
        default=0.0,
        metadata={"help": "The ratio of GPU remaining memory (after the first forward-backward)" \
                          "which will be used when using hybrid CPU optimizer. This argument is meaningless " \
                          "when `tensor_placement_policy` of `ShardedModelV2` is not 'auto'. Defaults to 0.0."}
    )


@dataclass
class DatasetConfig(HulkDataclass):
    num_workers: int = field(
        default=1, metadata={"help": "how many subprocesses to use for data loading"}
    )
    skip_invalid_size_inputs_valid_test: bool = field(
        default=False,
        metadata={"help": "ignore too long or too short lines in valid and test set"},
    )
    max_tokens: Optional[int] = field(
        default=None, metadata={"help": "maximum number of tokens in a batch"}
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "number of examples in a batch",
            "argparse_alias": "--max-sentences",
        },
    )
    required_batch_size_multiple: int = field(
        default=8, metadata={"help": "batch size will be a multiplier of this value"}
    )
    required_seq_len_multiple: int = field(
        default=1,
        metadata={
            "help": "maximum sequence length in batch will be a multiplier of this value"
        },
    )
    dataset_impl: Optional[DATASET_IMPL_CHOICES] = field(
        default=None, metadata={"help": "output dataset implementation"}
    )
    data_buffer_size: int = field(
        default=10, metadata={"help": "Number of batches to preload"}
    )
    train_subset: str = field(
        default="train",
        metadata={"help": "data subset to use for training (e.g. train, valid, test)"},
    )
    valid_subset: str = field(
        default="valid",
        metadata={
            "help": "comma separated list of data subsets to use for validation"
            " (e.g. train, valid, test)"
        },
    )
    combine_valid_subsets: Optional[bool] = field(
        default=None,
        metadata={
            "help": "comma separated list of data subsets to use for validation"
            " (e.g. train, valid, test)",
            "argparse_alias": "--combine-val",
        },
    )
    ignore_unused_valid_subsets: Optional[bool] = field(
        default=False,
        metadata={"help": "do not raise error if valid subsets are ignored"},
    )

    validate_interval: int = field(
        default=1, metadata={"help": "validate every N epochs"}
    )
    validate_interval_updates: int = field(
        default=0, metadata={"help": "validate every N updates"}
    )
    validate_after_updates: int = field(
        default=0, metadata={"help": "dont validate until reaching this many updates"}
    )
    fixed_validation_seed: Optional[int] = field(
        default=None, metadata={"help": "specified random seed for validation"}
    )
    disable_validation: bool = field(
        default=False, metadata={"help": "disable validation"}
    )
    max_tokens_valid: Optional[int] = field(
        default=II("dataset.max_tokens"),
        metadata={
            "help": "maximum number of tokens in a validation batch"
            " (defaults to --max-tokens)"
        },
    )
    batch_size_valid: Optional[int] = field(
        default=II("dataset.batch_size"),
        metadata={
            "help": "batch size of the validation batch (defaults to --batch-size)",
            "argparse_alias": "--max-sentences-valid",
        },
    )
    max_valid_steps: Optional[int] = field(
        default=None,
        metadata={"help": "How many batches to evaluate", "argparse_alias": "--nval"},
    )
    curriculum: int = field(
        default=0, metadata={"help": "don't shuffle batches for first N epochs"}
    )
    gen_subset: str = field(
        default="test",
        metadata={"help": "data subset to generate (train, valid, test)"},
    )
    num_shards: int = field(
        default=1, metadata={"help": "shard generation over N shards"}
    )
    shard_id: int = field(
        default=0, metadata={"help": "id of the shard to generate (id < num_shards)"}
    )
    grouped_shuffling: bool = field(
        default=False,
        metadata={
            "help": "shuffle batches in groups of num_shards to enable similar sequence lengths on each GPU worker when batches are sorted by length",
        },
    )
    update_epoch_batch_itr: bool = field(
        default=II("dataset.grouped_shuffling"),
        metadata={
            "help": "if true then prevents the reuse the epoch batch iterator by setting can_reuse_epoch_itr to false, defaults to --grouped-shuffling )",
        },
    )
    update_ordered_indices_seed: bool = field(
        default=False,
        metadata={
            "help": "if true then increment seed with epoch for getting batch iterators, defautls to False.",
        },
    )


@dataclass
class OptimizationConfig(HulkDataclass):
    max_epoch: int = field(
        default=0, metadata={"help": "force stop training at specified epoch"}
    )
    max_update: int = field(
        default=0, metadata={"help": "force stop training at specified update"}
    )
    stop_time_hours: float = field(
        default=0,
        metadata={
            "help": "force stop training after specified cumulative time (if >0)"
        },
    )
    clip_norm: float = field(
        default=0.0, metadata={"help": "clip threshold of gradients"}
    )
    grad_norm_threashold: float = field(
        default=float("inf"), metadata={"help": "Current batch data will be skipped when grad norm greater than grad_norm_threashold"}
    )
    clip_norm_type: Optional[CLIP_GRAD_NORM_TYPE_CHOICES] = field(
        default="l2",
        metadata={"help": "either 'l2' or 'inf' to clip by l2 norm or max abs grad"},
    )
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
    )
    update_freq: List[int] = field(
        default_factory=lambda: [1],
        metadata={"help": "update parameters every N_i batches, when in epoch i"},
    )
    lr: List[float] = field(
        default_factory=lambda: [0.25],
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    stop_min_lr: float = field(
        default=-1.0,
        metadata={"help": "stop training when the learning rate reaches this minimum"},
    )
    use_bmuf: bool = field(
        default=False,
        metadata={
            "help": "specify global optimizer for syncing models on different GPUs/shards"
        },
    )
    skip_remainder_batch: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, include the last (partial) batch of each epoch in training"
            " (default is to skip it)."
        },
    )


@dataclass
class CheckpointConfig(HulkDataclass):
    save_dir: str = field(
        default="checkpoints", metadata={"help": "path to save checkpoints"}
    )
    restore_file: str = field(
        default="checkpoint-last",
        metadata={
            "help": "filename from which to load checkpoint "
            "(default: <save-dir>/checkpoint-last"
        },
    )
    finetune_from_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "finetune from a pretrained model; note that meters and lr scheduler will be reset"
        },
    )
    reset_dataloader: bool = field(
        default=False,
        metadata={
            "help": "if set, does not reload dataloader state from the checkpoint"
        },
    )
    reset_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": "if set, does not load lr scheduler state from the checkpoint"
        },
    )
    reset_meters: bool = field(
        default=False,
        metadata={"help": "if set, does not load meters from the checkpoint"},
    )
    reset_optimizer: bool = field(
        default=False,
        metadata={"help": "if set, does not load optimizer state from the checkpoint"},
    )
    optimizer_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override optimizer args when loading a checkpoint"
        },
    )
    save_interval: int = field(
        default=1, metadata={"help": "save a checkpoint every N epochs"}
    )
    save_interval_updates: int = field(
        default=0, metadata={"help": "save a checkpoint (and validate) every N updates"}
    )
    enforce_save_interval_updates: int = field(
        default=0, metadata={"help": "enforce save a last checkpoint (no validate) every N updates"}
    )
    keep_interval_updates: int = field(
        default=-1,
        metadata={
            "help": "keep the last N checkpoints saved with --save-interval-updates"
        },
    )
    keep_interval_updates_pattern: int = field(
        default=-1,
        metadata={
            "help": "when used with --keep-interval-updates, skips deleting "
            "any checkpoints with update X where "
            "X %% keep_interval_updates_pattern == 0"
        },
    )
    keep_last_epochs: int = field(
        default=-1, metadata={"help": "keep last N epoch checkpoints"}
    )
    keep_last_parts: int = field(
        default=-1, metadata={"help": "keep last N part checkpoints"}
    )
    keep_best_checkpoints: int = field(
        default=-1, metadata={"help": "keep best N checkpoints based on scores"}
    )
    no_save: bool = field(
        default=False, metadata={"help": "don't save models or checkpoints"}
    )
    no_part_checkpoints: bool = field(
        default=True, metadata={"help": "don't save models when current part finished"}
    )
    no_epoch_checkpoints: bool = field(
        default=False, metadata={"help": "only store last and best checkpoints"}
    )
    no_last_checkpoints: bool = field(
        default=False, metadata={"help": "don't store last checkpoints"}
    )
    no_save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "don't save optimizer-state as part of checkpoint"},
    )
    best_checkpoint_metric: str = field(
        default="loss", metadata={"help": 'metric to use for saving "best" checkpoints'}
    )
    maximize_best_checkpoint_metric: bool = field(
        default=False,
        metadata={
            "help": 'select the largest metric value for saving "best" checkpoints'
        },
    )
    patience: int = field(
        default=-1,
        metadata={
            "help": (
                "early stop training if valid performance doesn't "
                "improve for N consecutive validation runs; note "
                "that this is influenced by --validate-interval"
            )
        },
    )
    checkpoint_suffix: str = field(
        default="", metadata={"help": "suffix to add to the checkpoint file name"}
    )
    checkpoint_shard_count: int = field(
        default=1,
        metadata={
            "help": "Number of shards containing the checkpoint - "
            "if the checkpoint is over 300GB, it is preferable "
            "to split it into shards to prevent OOM on CPU while loading "
            "the checkpoint"
        },
    )
    io_strategy: CKPT_IO_STRATEGY = field(
        default='greedy_balance',
        metadata={
            "help": "Checkpoints io strategy on all devices, options: [default, master, greedy_balance, balance]"
        }
    )
    barrier_load_checkpoint: bool = field(
        default=True,
        metadata={
            "help": "If set, synchronizes all processes by default TCP store to wait loading checkpoint."
        }
    )
    skip_num_batches: int = field(
        default=0,
        metadata={
            "help": "if set, skip N batches when loading checkpoint"
        },
    )
    saving_sync: bool = field(
        default=True,
        metadata={"help": "If set, synchronize file saving to ensure consistency of checkpoint files."}
    )


@dataclass
class GenerationConfig(HulkDataclass):
    use_beam_search: bool = field(
        default=False,
        metadata={"help": "use beam search if set to True"},
    )
    beam: int = field(
        default=5,
        metadata={"help": "beam size"},
    )
    max_batch_size: int = field(
        default=20,
        metadata={"help": "Maximum batchsize for single inference"},
    )
    inference_batch_times_seqlen_threshold: int = field(
        default=512,
        metadata={"help": "During inference, if batch-size times "
                       "sequence-length is smaller than this threshold "
                       "then we will not use pipelining, otherwise we will."},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "number of hypotheses to output"},
    )
    max_len_a: float = field(
        default=0,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    max_len_b: int = field(
        default=200,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    min_len: int = field(
        default=1,
        metadata={"help": "minimum generation length"},
    )
    max_len: int = field(
        default=1,
        metadata={"help": "maximum generation length"},
    )
    match_source_len: bool = field(
        default=False,
        metadata={"help": "generations should match the source length"},
    )
    unnormalized: bool = field(
        default=False,
        metadata={"help": "compare unnormalized hypothesis scores"},
    )
    no_early_stop: bool = field(
        default=False,
        metadata={"help": "deprecated"},
    )
    no_beamable_mm: bool = field(
        default=False,
        metadata={"help": "don't use BeamableMM in attention layers"},
    )
    lenpen: float = field(
        default=1,
        metadata={
            "help": "length penalty: <1.0 favors shorter, >1.0 favors longer sentences"
        },
    )
    unkpen: float = field(
        default=0,
        metadata={
            "help": "unknown word penalty: <0 produces more unks, >0 produces fewer"
        },
    )
    replace_unk: Optional[str] = field(
        default=None,
        metadata={
            "help": "perform unknown replacement (optionally with alignment dictionary)",
            "argparse_const": "@@ ",
        },
    )
    sacrebleu: bool = field(
        default=False,
        metadata={"help": "score with sacrebleu"},
    )
    score_reference: bool = field(
        default=False,
        metadata={"help": "just score the reference translation"},
    )
    prefix_size: int = field(
        default=0,
        metadata={"help": "initialize generation by target prefix of given length"},
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": "ngram blocking such that this size ngram cannot be repeated in the generation"
        },
    )
    repeat_penalty: float = field(
        default=1.2,
        metadata={
            "help": "the fixed penalty value of repeated tokens."
        }
    )
    num_repeat_penalty: float = field(
        default=0.1,
        metadata={
            "help": "the penalty value for the number of repetitions of repeated tokens."
        }
    )
    sampling: bool = field(
        default=False,
        metadata={"help": "sample hypotheses instead of using beam search"},
    )
    sampling_topk: int = field(
        default=-1,
        metadata={"help": "sample from top K likely next words instead of all words"},
    )
    sampling_topp: float = field(
        default=-1.0,
        metadata={
            "help": "sample from the smallest set whose cumulative probability mass exceeds p for next words"
        },
    )
    constraints: Optional[GENERATION_CONSTRAINTS_CHOICES] = field(
        default=None,
        metadata={
            "help": "enables lexically constrained decoding",
            "argparse_const": "ordered",
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for generation"},
    )
    diverse_beam_groups: int = field(
        default=-1,
        metadata={"help": "number of groups for Diverse Beam Search"},
    )
    diverse_beam_strength: float = field(
        default=0.5,
        metadata={"help": "strength of diversity penalty for Diverse Beam Search"},
    )
    diversity_rate: float = field(
        default=-1.0,
        metadata={"help": "strength of diversity penalty for Diverse Siblings Search"},
    )
    print_alignment: Optional[PRINT_ALIGNMENT_CHOICES] = field(
        default=None,
        metadata={
            "help": "if set, uses attention feedback to compute and print alignment to source tokens "
            "(valid options are: hard, soft, otherwise treated as hard alignment)",
            "argparse_const": "hard",
        },
    )
    print_step: bool = field(
        default=False,
        metadata={"help": "print steps"},
    )
    lm_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to input prompts file"},
    )
    prompt_fmt: Optional[str] = field(
        default=None,
        metadata={"help": "the format string of query.such as ('%s<s>', '<User> %s<end><Bot> ', '问：%s<ret> <end>答：')."}
    )
    generate_times: int = field(
        default=1,
        metadata={"help": "The number of generation for each sample."}
    )
    lm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for lm probs for lm fusion"},
    )

    # arguments for iterative refinement generator
    iter_decode_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "if > 0.0, it penalized early-stopping in decoding."},
    )
    iter_decode_max_iter: int = field(
        default=10,
        metadata={"help": "maximum iterations for iterative refinement."},
    )
    iter_decode_force_max_iter: bool = field(
        default=False,
        metadata={
            "help": "if set, run exact the maximum number of iterations without early stop"
        },
    )
    iter_decode_with_beam: int = field(
        default=1,
        metadata={
            "help": "if > 1, model will generate translations varying by the lengths."
        },
    )
    iter_decode_with_external_reranker: bool = field(
        default=False,
        metadata={
            "help": "if set, the last checkpoint are assumed to be a reranker to rescore the translations"
        },
    )
    retain_iter_history: bool = field(
        default=False,
        metadata={
            "help": "if set, decoding returns the whole history of iterative refinement"
        },
    )
    retain_dropout: bool = field(
        default=False,
        metadata={"help": "Use dropout at inference time"},
    )
    # temporarily set to Any until https://github.com/facebookresearch/hydra/issues/1117 is fixed
    # retain_dropout_modules: Optional[List[str]] = field(
    retain_dropout_modules: Any = field(
        default=None,
        metadata={
            "help": "if set, only retain dropout for the specified modules; "
            "if not set, then dropout will be retained for all modules"
        },
    )
    # special decoding format for advanced decoding.
    decoding_format: Optional[GENERATION_DECODING_FORMAT_CHOICES] = field(
        default=None,
        metadata={"help": "special decoding format for advanced decoding."},
    )
    no_seed_provided: bool = field(
        default=False,
        metadata={"help": "if set, dont use seed for initializing random generators"},
    )
    dynamic_batch: bool = field(
        default=False,
        metadata={"help": "if set, batch data will rebuild when decode to eod"},
    )
    local_attention_memory_enable: bool = field(
        default=False,
        metadata={"help": "if set, max seq_len for k and v will be same as sparse_local_size"},
    )



@dataclass
class CommonEvalConfig(HulkDataclass):
    path: Optional[str] = field(
        default=None,
        metadata={"help": "path(s) to model file(s), colon separated"},
    )
    post_process: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "post-process text by removing BPE, letter segmentation, etc. "
                "Valid options can be found in fairseq.data.utils.post_process."
            ),
            "argparse_const": "subword_nmt",
            "argparse_alias": "--remove-bpe",
        },
    )
    quiet: bool = field(default=False, metadata={"help": "only print final scores"})
    model_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override model args at generation that were used during model training"
        },
    )
    results_path: Optional[str] = field(
        default=None, metadata={"help": "path to save eval results (optional)"}
    )


@dataclass
class EMAConfig(HulkDataclass):
    store_ema: bool = field(
        default=False, metadata={help: "store exponential moving average shadow model"}
    )
    ema_decay: float = field(
        default=0.9999, metadata={"help": "decay for exponential moving average model"}
    )
    ema_start_update: int = field(
        default=0, metadata={"help": "start EMA update after this many model updates"}
    )
    ema_seed_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Seed to load EMA model from. "
            "Used to load EMA model separately from the actual model."
        },
    )
    ema_update_freq: int = field(
        default=1, metadata={"help": "Do EMA update every this many model updates"}
    )
    ema_fp32: bool = field(
        default=False,
        metadata={"help": "If true, store EMA model in fp32 even if model is in fp16"},
    )


@dataclass
class LMDBDatasetConfig(DatasetConfig):
    drop_last: bool = field(
        default=False,
        metadata={"help": "Whether the last several samples would be drop after sharding the whole dataset to the cluster"}
    )
    enable_readbin: bool = field(
        default=False,
        metadata={"help": 
            "Each trained block's data will be ignored if this configuration is enabled"}
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether shuffle the order of the dataset"}
    )
    num_parts: int = field(
        default=1,
        metadata={"help": "how mamy parts the dataset split to"}
    )
    num_blocks: int = field(
        default=1,
        metadata={"help": "how many blocks to shuffle together"}
    )
    block_size: int = field(
        default=2000,
        metadata={"help": "how many samples reading from files per block"}
    )
    seed: int = field(
        default=1000,
        metadata={"help": "random seed for shuffle"}
    )
    load_type: str = field(
        default="PIPELINE",
        metadata={
            "help": "load type"
            " (e.g. ALL, PIPELINE, MODEL)"
        },
    )
    disable_cache: bool = field(
        default=False,
        metadata={"help": "turn off caching mechanism by block"}
    )
    disable_iterator_cache: bool = field(
        default=True,
        metadata={
            "help": "turn off caching mechanism of dataset iterator. If cache dataset iterator, " \
                    "will use dataset iterator built on first epoch, it may take up CPU memory " \
                    "for a long time, but don't need to repeatedly construct dataset iterator."
        }
    )
    train_ppo_subset: str = field(default='', metadata={'help': 'dataset path'})
    valid_ppo_subset: str = field(default='', metadata={'help': 'dataset path'})
    train_ptx_subset: str = field(default='', metadata={'help': 'dataset path'})
    valid_ptx_subset: str = field(default='', metadata={'help': 'dataset path'})
    prompt_mask: bool = field(
        default=True, metadata={'help': 'if set, will generate a prompt mask for sft training.'}
    )


@dataclass
class RLHFConfig(HulkDataclass):
    policy_model: Any = None
    critic_model: Any = None
    policy_criterion: Any = None
    critic_criterion: Any = None
    policy_optimizer: Any = None
    critic_optimizer: Any = None
    policy_lr_scheduler: Any = None
    critic_lr_scheduler: Any = None
    train_prompt_path: str = field(
        default="", metadata={"help": "prompts config path of train subset."}
    )
    valid_prompt_path: str = field(
        default="", metadata={"help": "prompts config path of valid subset."}
    )
    dl_num_workers: int = field(
        default=0, metadata={"help": "num workers of dataloader."}
    )
    max_episode: int = field(
        default=10, metadata={"help": "max episode of rlhf training."}
    )
    max_ppo_epoch: int = field(
        default=1, metadata={"help": "num epoch on one episode."}
    )
    generate_engine_enable: bool = field(
        default=True,
        metadata={"help": "if set, using generate engine to generate actions."},
    )
    sampling_batch_size: int = field(
        default=64, metadata={"help": "batch size of sampling."}
    )
    infer_batch_size: int = field(
        default=64, metadata={"help": "batch size of get log probs."}
    )
    replay_buffer_save_internal: int = field(
        default=5000, metadata={"help": "replay buffer save internal."}
    )
    update_timesteps: int = field(
        default=1000, metadata={"help": "update time steps of ppo training."}
    )
    critic_max_tokens: int = field(
        default=1024, metadata={"help": "max tokens per batch of cirtic model."}
    )
    policy_max_tokens: int = field(
        default=1024, metadata={"help": "max tokens per batch of policy model."}
    )
    pretrain_batch_size: int = field(
        default=1, metadata={"help": "batch size of pretrain dataset."}
    )
    critic_num_micro_batch: int = field(
        default=1, metadata={"help": "num micro batch of critic model."}
    )
    policy_num_micro_batch: int = field(
        default=1, metadata={"help": "num micro batch of policy model."}
    )
    dump_validate_output: bool = field(
        default=True, metadata={"help": "if set, will dump validate output to file."}
    )
    max_query_length: int = field(
        default=1024,
        metadata={
            "help": "if set, prompts whoes lengths > `max_query_length` are ignored."
        },
    )
    add_user_bot: bool = field(
        default=True,
        metadata={"help": "if set, query will format by `<User> %s<end><Bot> `."},
    )
    check_logits_overflow: bool = field(
        default=False,
        metadata={
            "help": "if set, generate engine will check whether the logits overflows at each step."
        },
    )
    critic_zero_centering: bool = field(
        default=True,
        metadata={
            "help": "if set, reward score and values will subtract 0.5 to shift its mean to 0."
        },
    )
    # hyper parameters of computes rewards/values/advantages/returns
    use_whiten_reward: bool = field(
        default=False,
        metadata={'help': 'use whiten reward'},
    )
    global_reward_statistics: bool = field(
        default=True,
        metadata={'help': 'use global reward statistics to whiten reward.'},
    )
    use_reward_norm: bool = field(
        default=True,
        metadata={'help': 'use standardization to normalize rewards.'},
    )
    use_whiten_advantages: bool = field(
        default=False,
        metadata={'help': "use whiten advantages."}
    )
    kl_coef: float = field(
        default=0.02, metadata={"help": "kl coef for computing rewards."}
    )
    adap_kl_ctrl: bool = field(
        default=False, metadata={"help": "Use adaptive KL control, otherwise linear"}
    )
    adap_kl_ctrl_target: float = field(
        default=6, metadata={"help": "Target KL value for adaptive KL control"}
    )
    adap_kl_ctrl_horizon: float = field(
        default=10000, metadata={"help": "Horizon for adaptive KL control"}
    )
    clip_reward_value: float = field(
        default=5, metadata={"help": "clip value to control reward distribution."}
    )
    gamma: float = field(
        default=1.0,
        metadata={
            "help": "`gamma` is used to calculate the advantages when using GAE method."
        },
    )
    lam: float = field(
        default=0.95,
        metadata={
            "help": "`lam` is used to calculate the advantages when using GAE method."
        },
    )
    ptx_gamma: float = field(
        default=0.9,
        metadata={
            "help": "pretrain loss factor of rlhf criterion."
        }
    )


@dataclass
class LoRAConfig(HulkDataclass):
    apply_lora: bool = field(
        default=False,
        metadata={
            "help": "fine-tune the model with LoRA if enabled."
        }
    )
    lora_rank: int = field(
        default=0, metadata={"help": "lora hidden layer dimension"}
    )
    lora_alpha: int = field(
        default=128, metadata={"help": "lora attn alpha"}
    )
    lora_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for lora layers"}
    )
    adapt_q: bool = field(
        default=False, metadata={"help": "adapting the attention query weights"}
    )
    adapt_k: bool = field(
        default=False, metadata={"help": "adapting the attention key weights"}
    )
    adapt_v: bool = field(
        default=False, metadata={"help": "adapting the attention value weights"}
    )
    adapt_o: bool = field(
        default=False, metadata={"help": "adapting the attention output project weights"}
    )
    adapt_fc1: bool = field(
        default=False, metadata={"help": "adapting the first linear in FFN module"}
    )
    adapt_fc2: bool = field(
        default=False, metadata={"help": "adapting the first second in FFN module"}
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "merge two branches during evaluation if enabled"}
    )
    from_pretrained: Optional[str] = field(
        default=None,
        metadata={
            "help": "Load pretrained LoRA model if from_pretrained is not None"
        }
    )


@dataclass
class KDConfig(HulkDataclass):
    teacher_output_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "teacher output path"
        }
    )
    topk: int = field(
        default=1, metadata={"help": "top k value"}
    )
    kl_loss_facor: float = field(
        default=1.0, metadata={"help": "KL loss factor"}
    )

@dataclass
class QuantConfig(HulkDataclass):
    apply_quant: bool = field(
        default=False,
        metadata={
            "help": "Quant the model if enabled."
        }
    )
    calibration_sample_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path of the calibration sample"
        },
    )
    quant_scales_path: Optional[str] = field(
        default="quant_scales",
        metadata={
            "help": "path to static quantized activation maximum value"
        },
    )
    channel_msg_dir: Optional[str] = field(
        default="channel_msg_dir",
        metadata={
            "help": "directory to activation channel msg"
        },
    )
    weight_quant: Optional[WEIGHT_QUANT_STRATEGY] = field(
        default=None,
        metadata={
            "help": "weight quantization method, you can choose per_tensor, per_token and other (do nothing)"
        },
    )
    act_quant: Optional[ACT_QUANT_STRATEGY] = field(
        default=None,
        metadata={
            "help": "activation quantization method, you can choose static_per_tensor, per_tensor, per_token and other (do nothing)"
        },
    )
    quantize_output: bool = field(
        default=False,
        metadata={
            "help": "used to select whether to quantize the output. The quantized output can be used for subsequent BMM."
        }
    )
    quant_layers: List[str] = field(
        default_factory=lambda: ['qkv_proj', 'out_proj', 'fc1', 'fc2'],
        metadata={
            "help": "List of names of the layers to be quantized"
        },
    )
    clamp_layers: List[str] = field(
        default_factory=lambda: ['fc2'],
        metadata={
            "help": "List of names of the layers to be clamp"
        },
    )
    fc2_ignore_layers: List[int] = field(
        default_factory=lambda: [],
        metadata={
            "help": "used to ignore which FC2 layers do not require quantization."
        }
    )
    clamp_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "num for clamp"
        },
    )
    percentile: float = field(
        default=1.0,
        metadata={
            "help": "set the percentage to obtain the maximum value"
        },
    )
    save_quant_model: bool = field(
        default=False,
        metadata={
            "help": "used to select whether to save quant model."
        }
    )
    save_dir: str = field(
        default="quant_model", metadata={"help": "path to save checkpoints"}
    )
    data_bits: Optional[int] = field(
        default=None, metadata={"help": "input quant bits"}
    )
    parameter_bits: Optional[int] = field(
        default=None, metadata={"help": "parameter quant bits"}
    )
    quant_mode: bool = field(
        default=True, metadata={"help": "quant mode setting, including: MaxValue and QValue, default MaxValue quant"}
    )
    extra: Any = None

@dataclass
class ClampConfig(HulkDataclass):
    apply_clamp: bool = field(
        default=False,
        metadata={
            "help": "fine-tune the model with clamp if enabled."
        }
    )

    clamp_q_k_v_proj_input: Optional[float] = field(
        default=None, metadata={"help": "clamp q_k_v_proj input"}
    )
    clamp_q_k_v_proj_weight: Optional[float] = field(
        default=None, metadata={"help": "clamp q_k_v_proj weight"}
    )

    clamp_out_proj_input: Optional[float] = field(
        default=None, metadata={"help": "clamp out_proj input"}
    )
    clamp_out_proj_weight: Optional[float] = field(
        default=None, metadata={"help": "clamp out_proj weight"}
    )

    clamp_fc1_input: Optional[float] = field(
        default=None, metadata={"help": "clamp fc1 input"}
    )
    clamp_fc1_weight: Optional[float] = field(
        default=None, metadata={"help": "clamp fc1 weight"}
    )

    clamp_fc2_input: Optional[float] = field(
        default=None, metadata={"help": "clamp fc2 input"}
    )
    clamp_fc2_weight: Optional[float] = field(
        default=None, metadata={"help": "clamp fc2 weight"}
    )


@dataclass
class HulkConfig(HulkDataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    model_parallel: ModelParallelConfig = ModelParallelConfig()
    zero: ZeroConfig = ZeroConfig()
    dataset: LMDBDatasetConfig = LMDBDatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    generation: GenerationConfig = GenerationConfig()
    lora: LoRAConfig = LoRAConfig()
    model: Any = None
    task: Any = None
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    tokenizer: Any = None
    arch: Any = None
    decoder: Any = None
    ema: EMAConfig = EMAConfig()
    rlhf: RLHFConfig = RLHFConfig()
    kd: KDConfig = KDConfig()
    quant: QuantConfig = QuantConfig()
    clamp: ClampConfig = ClampConfig()

