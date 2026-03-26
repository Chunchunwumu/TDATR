import os
import random

import torch
from omegaconf import open_dict

def infer_init_method(cfg, force_distributed=False):
    if cfg.distributed_init_method is not None:
        return
    
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        # support torch.distributed.launch
        _infer_torch_distributed_launch_init(cfg)
    elif cfg.distributed_world_size >= 1 or force_distributed:
        # fallback for single node with multiple GPUs
        _infer_single_node_init(cfg)

    elif not cfg.distributed_no_spawn:
        with open_dict(cfg):
            cfg.distributed_num_procs = min(
                torch.cuda.device_count(), cfg.distributed_world_size
            )


def _infer_torch_distributed_launch_init(cfg):
    cfg.distributed_init_method = "env://"
    cfg.distributed_world_size = int(os.environ["WORLD_SIZE"])
    cfg.distributed_rank = int(os.environ["RANK"])
    cfg.distributed_master_addr = os.environ['MASTER_ADDR']
    cfg.distributed_master_port = int(os.environ['MASTER_PORT'])
    # processes are created by torch.distributed.launch
    cfg.distributed_no_spawn = True


def _infer_single_node_init(cfg):
    assert (
        cfg.distributed_world_size <= torch.cuda.device_count()
    ), f"world size is {cfg.distributed_world_size} but have {torch.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    cfg.distributed_init_method = "tcp://localhost:{port}".format(port=port)
    cfg.distributed_master_addr = 'localhost'
    cfg.distributed_master_port = port


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



def call_main(cfg, main, **kwargs):
    if cfg.distributed_training.distributed_init_method is None:
        infer_init_method(cfg.distributed_training)
    if cfg.distributed_training.distributed_init_method is not None:
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