# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import logging
from hydra.core.config_store import ConfigStore
from TDATR_utils.dataclass import HulkConfig
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def _hijack_hydra_configure_logging():
    import sys
    import hydra
    from typing import Union, Sequence

    def _configure_log(
        log_config: DictConfig,
        verbose_config: Union[bool, str, Sequence[str]] = False,
    ) -> None:
        assert isinstance(verbose_config, (bool, str)) or OmegaConf.is_list(verbose_config)
        if log_config is not None:
            conf: Dict[str, Any] = OmegaConf.to_container(  # type: ignore
                log_config, resolve=True
            )
            if conf["root"] is not None:
                file_name = conf.get('handlers', {}).get('file', {}).get('filename')
                conf['handlers']['file']['filename'] = file_name + ".tempfile"
                logging.config.dictConfig(conf)
        else:
            # default logging to stdout
            root = logging.getLogger()
            root.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
            )
            handler.setFormatter(formatter)
            root.addHandler(handler)
        if isinstance(verbose_config, bool):
            if verbose_config:
                logging.getLogger().setLevel(logging.DEBUG)
        else:
            if isinstance(verbose_config, str):
                verbose_list = OmegaConf.create([verbose_config])
            elif OmegaConf.is_list(verbose_config):
                verbose_list = verbose_config  # type: ignore
            else:
                assert False

            for logger in verbose_list:
                logging.getLogger(logger).setLevel(logging.DEBUG)

    def _try(fn, src_fn):
        def exec_fn(*args, **kwargs):
            try:
                ret = fn(*args, **kwargs)
            except Exception as e:
                print(f'error: {e}')
                ret = src_fn(*args, **kwargs)
            return ret
        return exec_fn

    org_configure_log_fn = hydra.core.utils.configure_log
    hydra.core.utils.configure_log = _try(_configure_log, org_configure_log_fn)


def hydra_init(cfg_name="config") -> None:
    _hijack_hydra_configure_logging()
    cs = ConfigStore.instance()
    cs.store(name=f"{cfg_name}", node=HulkConfig)
    
    for k in HulkConfig.__dataclass_fields__:
        v = HulkConfig.__dataclass_fields__[k].default
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise



import os
import logging
import socket
import time
from argparse import Namespace

import torch
from torch.backends import cudnn
import torch.distributed as dist

from TDATR_utils.global_context import global_context as gpc
from TDATR_utils.global_variables import ParallelMode


logger = logging.getLogger(__name__)


def _check_model_parallel_topo(cfg: HulkConfig) -> None:
    # Distributed args.
    dist_cfg = cfg.distributed_training
    mpu_cfg = cfg.model_parallel
    world_size = dist_cfg.distributed_world_size
    pp_size = mpu_cfg.pipeline_model_parallel_size
    tp_size = mpu_cfg.tensor_model_parallel_size
    sp_size = mpu_cfg.sequence_parallel_size
    mp_size = pp_size * tp_size
    
    assert mp_size * sp_size <= world_size, \
        f"world size ({world_size}) is less than `tensor model parallel size({tp_size}) * " \
        f"pipeline model parallel size({pp_size}) * sequence parallel size({sp_size})`!" \
    
    assert world_size % (mp_size * sp_size) == 0, \
        f"world size({world_size}) is not divisible by `model parallel size({mp_size}) * sequence parallel size({sp_size})`"
    
    mpu_cfg.data_parallel_size = world_size // mp_size // sp_size
    if (not torch.cuda.is_available() or cfg.common.cpu):
        virtual_pp_size = mpu_cfg.virtual_pipeline_model_parallel_size
        assert (mp_size == 1 and virtual_pp_size is None), \
            f"model parallelism isn't supported using cpu traing or cuda is unavailable."


def _check_model_parallel_cfgs(cfg: HulkConfig) -> None:
    dist_cfg = cfg.distributed_training
    mpu_cfg = cfg.model_parallel
    assert mpu_cfg.micro_batch_size >= 1, \
        f"Required micro_batch_size >= 1, but got {mpu_cfg.micro_batch_size}."
    if mpu_cfg.global_batch_size is None and mpu_cfg.num_micro_batch is not None:
        assert mpu_cfg.micro_batch_size > 0 and mpu_cfg.num_micro_batch > 0, \
                        f"Micro_batch_size({mpu_cfg.micro_batch_size}) <= 0 " \
                        f"or num_micro_batch({mpu_cfg.num_micro_batch}) <= 0"
        mpu_cfg.global_batch_size = mpu_cfg.micro_batch_size * \
                                    mpu_cfg.num_micro_batch * \
                                    mpu_cfg.data_parallel_size
    elif mpu_cfg.global_batch_size is not None and mpu_cfg.num_micro_batch is None:
        mpu_cfg.num_micro_batch = mpu_cfg.global_batch_size / \
                                    mpu_cfg.data_parallel_size / \
                                    mpu_cfg.micro_batch_size
    elif mpu_cfg.global_batch_size is not None and mpu_cfg.num_micro_batch is not None:
        global_batch_size_2 = mpu_cfg.micro_batch_size * \
                                mpu_cfg.num_micro_batch * \
                                mpu_cfg.data_parallel_size
        assert mpu_cfg.global_batch_size == global_batch_size_2, \
                        f"Required global_batch_size == micro_batch_size * num_micro_batch * data_parallel_size, " \
                        f"bug got global_batch_size = {mpu_cfg.global_batch_size} " \
                        f"and micro_batch_size * num_micro_batch * data_parallel_size = {global_batch_size_2}"
    else:
        raise ValueError(f"Got global_batch_size and num_micro_batch are None,"
                            f" you need to set one of them to a valid value.")
    logger.warning(
        "Set batch_size({})=micro_batch_size({}) in model parallel training mode." \
        "".format(cfg.dataset.batch_size, mpu_cfg.micro_batch_size)
        )
    cfg.dataset.batch_size = mpu_cfg.micro_batch_size

    if mpu_cfg.pipeline_model_parallel_size != 1 and \
       cfg.distributed_training.ddp_backend in ["c10d", "pytorch_ddp"]:
        raise ValueError(f"c10d/pytorch_ddp is not supported when using pipeline parallel!")

    if mpu_cfg.virtual_pipeline_model_parallel_size is not None:
        if int(mpu_cfg.virtual_pipeline_model_parallel_size) <= 1:
            mpu_cfg.virtual_pipeline_model_parallel_size = None
            logger.warning(
                "Ignore invalid config of virtual_pipeline_model_parallel_size <= 1."
            )
        if cfg.task._name == "ipt_rlhf_task":
            policy_num_micro_batch = mpu_cfg.num_micro_batch+cfg.rlhf.policy_num_micro_batch
            assert policy_num_micro_batch % mpu_cfg.pipeline_model_parallel_size == 0, \
                'Number of microbatches+policy_num_micro_batch({}) is not divisible by pipeline-parallel ' \
                'size({}) when using interleaved schedule'.format(
                policy_num_micro_batch, mpu_cfg.pipeline_model_parallel_size)

            assert cfg.rlhf.critic_num_micro_batch % mpu_cfg.pipeline_model_parallel_size == 0, \
                'critic_num_micro_batch({}) is not divisible by pipeline-parallel ' \
                'size({}) when using interleaved schedule'.format(
                cfg.rlhf.critic_num_micro_batch, mpu_cfg.pipeline_model_parallel_size)
        else:
            assert mpu_cfg.num_micro_batch % mpu_cfg.pipeline_model_parallel_size == 0, \
                'Number of microbatches({}) is not divisible by pipeline-parallel ' \
                'size({}) when using interleaved schedule'.format(
                mpu_cfg.num_micro_batch, mpu_cfg.pipeline_model_parallel_size)

            if cfg.dataset.batch_size_valid != mpu_cfg.micro_batch_size:
                logger.warning(
                    "Detected batch_size_valid({}) != micro_batch_size({}) when using pipeline "
                    "interleaved schedule, we will set batch_size_valid=micro_batch_size, "
                    "if you mind this, don't pass --num-layers-per-virtual-pipeline-stage to "
                    "TDATR_utils.".format(cfg.dataset.batch_size_valid, mpu_cfg.micro_batch_size)
                )
                cfg.dataset.batch_size_valid = mpu_cfg.micro_batch_size

    logger.info(
        f'using global_batch_size: {mpu_cfg.global_batch_size}, ' \
        f'micro_batch_size: {mpu_cfg.micro_batch_size}, ' \
        f'num_micro_batches {mpu_cfg.num_micro_batch}'
    )

    # check sequence parallel
    if mpu_cfg.tensor_model_parallel_size == 1:
        mpu_cfg.sequence_parallel = False
        logger.info(
            "Disable sequence parallel when tensor parallel = 1 " \
            "to avoid change in numerics when sequence parallel is enabled"
        )
    
    # check async_tensor_model_parallel_allreduce
    if mpu_cfg.sequence_parallel:
        if mpu_cfg.async_tensor_model_parallel_allreduce:
            mpu_cfg.async_tensor_model_parallel_allreduce = False
            logger.info(
                "disable async_tensor_model_parallel_allreduce when " \
                "sequence parallel is enabled."
            )
        if mpu_cfg.scatter_gather_tensors_in_pipeline:
            mpu_cfg.scatter_gather_tensors_in_pipeline = False
            logger.info(
                "If sequence_parallel enabled, we should set " \
                "scatter_gather_tensors_in_pipeline to false" \
                "otherwise it will cause the transmitted tensor between pipeline stages to be destroyed."
            )

    # check ema
    assert not cfg.ema.store_ema, \
        "EMA is incompatible with model parallelism."
    
    # check update_freq
    assert tuple(cfg.optimization.update_freq) == (1,), \
        "Detected optimization.update_freq != [1], when using model parallel " \
        "traning mode, you should use --num-micro-batch, instead of --update-freq."
    cfg.optimization.update_freq = [mpu_cfg.num_micro_batch]

    assert not cfg.distributed_training.cpu_offload, "`cpu offload` is not supported!"
    assert not cfg.distributed_training.use_sharded_state, "`use_sharded_state` is not supported!"
    if all([
        cfg.distributed_training.ddp_backend == 'fully_sharded',
        cfg.distributed_training.zero_sharding == 'os'
    ]):
        raise ValueError("`fully_sharded` and `os`cannot be used at the same time")

    logger.info(
        f'using world size: {dist_cfg.distributed_world_size}, '\
        f'data-parallel-size: {mpu_cfg.data_parallel_size}, ' \
        f'sequence-parallel-size: {mpu_cfg.sequence_parallel_size}, ' \
        f'tensor-model-parallel size: {mpu_cfg.tensor_model_parallel_size}, ' \
        f'pipeline-model-parallel size: {mpu_cfg.pipeline_model_parallel_size}.')


def _check_other_cfgs(cfg: HulkConfig) -> None:
    if cfg.common.fp16 and cfg.common.bf16:
        raise ValueError(f"fp16 and bf16 can not both be True")
    
    if cfg.common.memory_efficient_bf16 and cfg.common.memory_efficient_fp16:
        raise ValueError(f"memory_efficient_bf16 and memory_efficient_fp16 can not both be True ")
    
    if (cfg.common.fp16 and cfg.common.memory_efficient_bf16) \
        or (cfg.common.bf16 and cfg.common.memory_efficient_fp16):
        raise ValueError(f"cfg.common.fp16(cfg.common.bf16) is only paired with" \
                         f"cfg.common.memory_efficient_fp16(cfg.common.memory_efficient_bf16) for use")

    if cfg.common.memory_efficient_fp16 and not cfg.common.fp16:
        cfg.common.fp16 = True
        logger.info(f"common.fp16 is enabled because cfg.common.memory_efficient_fp16 is True")

    if cfg.common.memory_efficient_bf16 and not cfg.common.bf16:
        cfg.common.bf16 = True
        logger.info(f"common.bf16 is enabled because cfg.common.memory_efficient_bf16 is True")

    if cfg.common.bf16:
        assert cfg.distributed_training.ddp_backend == "megatron_ddp" and \
            cfg.distributed_training.accumulate_allreduce_grads_in_fp32, \
                f"Please use *megatron_ddp* and enable accumulate_allreduce_grads_in_fp32 when training with bf16"
    
    if cfg.common.bf16 == False and cfg.common.fp16 == False:
        assert cfg.distributed_training.ddp_backend != "megatron_ddp", "`megatron_ddp` has been disabled in fp32 mode."

    assert cfg.common.fp16_no_flatten_grads == True, "`fp16_no_flatten_grads=false` has been disabled for use!"

    if cfg.common.amp and (cfg.common.fp16 or cfg.common.bf16):
        raise ValueError(f"fp16 and bf16 must be false when amp is enabeld")

    if cfg.common.fp32_residual_connection:
        assert cfg.common.fp16 or cfg.common.bf16, "residual connection in fp32 only supported when using fp16 or bf16."
    
    if cfg.lora.apply_lora:
        if cfg.model_parallel.tensor_model_parallel_size != 1:
            raise ValueError("LoRA does not support tensor parallelism.")
        if cfg.model.from_pretrained is None:
            raise ValueError("Loading pre-trained weights is necessary when using LoRA fine-tuning.")
        if "bias_dropout_fusion" in cfg.model and cfg.model.bias_dropout_fusion:
            cfg.model.bias_dropout_fusion = False
            logger.info("LoRA does not support bias_dropout_fusion")

    # if "using_streaming_iterator" in cfg.task and cfg.task.using_streaming_iterator:
    if getattr(cfg.task, "using_streaming_iterator", False):
        assert getattr(cfg.task, "offline_batches_path", None) is not None and \
            os.path.exists(cfg.task.offline_batches_path), \
            f"offline_batches_path({cfg.task.offline_batches_path} is not exist!)"
        assert cfg.dataset.num_parts == 1, f"num_parts can only be 1 when using streaming iterator"
    
def check_cfgs(cfg: HulkConfig) -> None:
    _check_model_parallel_topo(cfg)
    _check_model_parallel_cfgs(cfg)
    _check_other_cfgs(cfg)


def init_distributed(cfg: HulkConfig) -> None:
    # setting for debug, which enable the same initialization in the same machine
    cudnn.benchmark = True if cfg.common.cudnn_benchmark else False
    cudnn.deterministic = True if cfg.common.cudnn_deterministic else False
    cudnn.enabled = True if cfg.common.cudnn_enabled else False

    if isinstance(cfg, Namespace):
        from TDATR_utils.utils import convert_namespace_to_omegaconf

        cfg = convert_namespace_to_omegaconf(cfg)

    dist_cfg = cfg.distributed_training

    if torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed is already initialized, cannot initialize twice!"
        )
    assert torch.cuda.is_available()
    device = dist_cfg.distributed_rank % torch.cuda.device_count()
    if dist_cfg.distributed_local_rank is not None:
        assert dist_cfg.distributed_local_rank == device, \
            'expected local-rank to be the same as rank % device-count.'
    else:
        dist_cfg.distributed_local_rank = device
    dist_cfg.device_id = dist_cfg.distributed_local_rank

    logger.info(
        "distributed init (rank {}): {}".format(
            dist_cfg.distributed_rank,
            dist_cfg.distributed_init_method,
        )
    )
    torch.cuda.set_device(device)

    logger.warning(
        "[TDATR_utils_log] sleep success, distributed init (rank {}): {}".format(
            dist_cfg.distributed_rank,
            dist_cfg.distributed_init_method,
        )
    )
    # init global context
    gpc.config = cfg
    gpc.init_global_dist(
        rank=dist_cfg.distributed_rank,
        world_size=dist_cfg.distributed_world_size,
        backend=dist_cfg.distributed_backend,
        host=dist_cfg.distributed_master_addr,
        port=dist_cfg.distributed_master_port
    )

    logger.warning(
        "[TDATR_utils_log] init_global_dist success (rank {}): {}".format(
            dist_cfg.distributed_rank,
            dist_cfg.distributed_init_method,
        )
    )

    # gpc.set_device(device)
    gpc.init_parallel_groups()
    # gather all workers using GLOO group
    gpc.gather_workers()

    logger.info(
        "initialized host {} as rank {}".format(
            socket.gethostname(),
            dist_cfg.distributed_rank,
        )
    )

    # perform a dummy all-reduce to initialize the NCCL communicator
    if torch.cuda.is_available():
        logger.info(
            "{} | {}/{} | Perform a dummy all-reduce to initialize the NCCL communicator".format(
                socket.gethostname(), dist_cfg.distributed_rank, dist_cfg.distributed_world_size
            )
        )
        dist.all_reduce(
            torch.zeros(gpc.get_world_size(ParallelMode.GLOBAL)).cuda(),
            group=gpc.get_group(ParallelMode.GLOBAL))
        if gpc.get_world_size(ParallelMode.DATA) > 1:
            dist.all_reduce(
                torch.zeros(gpc.get_world_size(ParallelMode.DATA)).cuda(),
                group=gpc.get_group(ParallelMode.DATA))
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            dist.all_reduce(
                torch.zeros(gpc.get_world_size(ParallelMode.TENSOR)).cuda(),
                group=gpc.get_group(ParallelMode.TENSOR))
        if gpc.get_world_size(ParallelMode.PIPELINE) > 1:
            dist.all_reduce(
                torch.zeros(gpc.get_world_size(ParallelMode.PIPELINE)).cuda(),
                group=gpc.get_group(ParallelMode.PIPELINE))
        logger.info(
            "{} | {}/{} | NCCL communicator initialization succeeded!".format(
                socket.gethostname(), dist_cfg.distributed_rank, dist_cfg.distributed_world_size
            )
        )

    dist_cfg.distributed_rank = torch.distributed.get_rank()

    # init rng seed manager
    gpc.set_seed(cfg.common.seed)
    _set_jit_fusion_options()


def _set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        #torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)


def initialize_logging(cfg: OmegaConf) -> None:
    file_name = None
    job_logging_cfg = None
    use_hydra = HydraConfig.initialized() or "job_logging_cfg" in cfg

    if use_hydra:
        if HydraConfig.initialized():
            hydra_config = HydraConfig.get()
            cfg.common.experiment_dir = \
                hydra_config.get("run", {}).get("dir", "./")
        if "job_logging_cfg" in cfg:
            job_logging_cfg = OmegaConf.to_container(cfg.job_logging_cfg)
            file_name = (
                job_logging_cfg
                .get('handlers', {})
                .get('file', {})
                .get('filename', None)
            )
            if file_name is not None:
                cfg.common.log_file = file_name
    else:
        file_name = cfg.common.log_file

    rank = dist.get_rank() if dist.is_initialized() else 0

    # if not using hydra, save config.yaml to exp dir.
    if not use_hydra: 
        os.makedirs(cfg.common.experiment_dir, exist_ok=True)
        os.chdir(cfg.common.experiment_dir)
        if rank == 0:
            cfg_file = "config.yaml"
            if os.path.exists(cfg_file):
                _backup(cfg_file)
            OmegaConf.save(cfg, os.path.join(cfg.common.experiment_dir, cfg_file))

    # setup log file handler
    if file_name is not None:
        # backup history log file
        if rank == 0 and os.path.exists(file_name):
            _backup(file_name)
            assert not os.path.exists(file_name)
        # Wait for backup to be completed
        if dist.is_initialized():
            dist.barrier()

        if use_hydra and job_logging_cfg is not None:
            # first, copy temp log file to dist log file
            temp_file = file_name + ".tempfile"
            if rank == 0 and os.path.exists(temp_file):
                shutil.copyfile(temp_file, file_name)

            # then, remove source file handler(temp file)
            src_file_handler = None
            for handler in logging.root.handlers:
                if (
                    isinstance(handler, logging.FileHandler) \
                    and \
                    temp_file in handler.baseFilename
                ):
                    src_file_handler = handler
                    break

            if src_file_handler is not None:
                logging.root.removeHandler(src_file_handler)
            
            # finally, add new file handler to logging
            # add new file handler to logging from `hydra`
            job_logging_cfg['handlers']['file']['filename'] = file_name
            logging.config.dictConfig(job_logging_cfg)

            if dist.is_initialized():
                dist.barrier()

            # remove source temp log file
            if rank == 0 and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        else:
            # add file handler from `cfg.common.log_file`
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            datefmt="%Y-%m-%d %H:%M:%S"
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(getattr(logging, 'INFO'))
            formatter = logging.Formatter(format, datefmt=datefmt)
            file_handler.setFormatter(formatter)
            logging.root.addHandler(file_handler)
        
    
     # set stream/file handler level, only master's level set to `INFO`
    for handler in logging.root.handlers:
        level = logging.INFO if rank == 0 else logging.WARNING
        if (
            isinstance(handler, logging.StreamHandler) \
                and \
            handler.stream.name in {'<stdout>', '<stderr>'}
        ):
            handler.setLevel(level)
        if (
            isinstance(handler, logging.FileHandler) \
            and \
            file_name is not None and file_name in handler.baseFilename
        ):
            handler.setLevel(level)

    if file_name is not None and not cfg.common.disable_dist_logging:
        # Back up log files with the same name
        os.makedirs("logs", exist_ok=True)
        log_name, ext = os.path.splitext(os.path.basename(file_name))
        dist_log_file = os.path.join("logs", f"{log_name}-rank#{rank}{ext}")
        if os.path.exists(dist_log_file):
            _backup(dist_log_file)
            assert not os.path.exists(dist_log_file)

        # add dist logging file handler
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        datefmt="%Y-%m-%d %H:%M:%S"
        file_handler = logging.FileHandler(dist_log_file)
        file_handler.setLevel(getattr(logging, 'INFO'))
        formatter = logging.Formatter(format, datefmt=datefmt)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

def initialize_hulk(cfg: HulkConfig) -> None:
    logger.warning(
        "[TDATR_utils_log] host {} | rank {} has been started!".format(
            socket.gethostname(),
            cfg.distributed_training.distributed_rank,
        )
    )
    check_cfgs(cfg)
    init_distributed(cfg)
    # initialize_logging(cfg)
