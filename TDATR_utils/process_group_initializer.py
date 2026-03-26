#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
from typing import Dict, Optional, Tuple, Any
import torch.distributed as dist
from omegaconf import DictConfig
from abc import ABC, abstractmethod
import logging
import time
from datetime import timedelta
from typing import Union, Sequence
import socket
try:
    from torch_npu.distributed import distributed_c10d
    from torch_npu.distributed.distributed_c10d import (
        barrier,
        Backend,
        GroupMember,
        get_backend,
        default_pg_timeout,
        _get_default_group,
        _new_process_group_helper,
        STORE_BASED_BARRIER_PREFIX,
    )
except:
    from torch.distributed import distributed_c10d
    from torch.distributed.distributed_c10d import (
        barrier,
        Backend,
        GroupMember,
        get_backend,
        default_pg_timeout,
        _get_default_group,
        _new_process_group_helper,
        STORE_BASED_BARRIER_PREFIX,
    )
from TDATR_utils.global_variables import ParallelMode

import torch.distributed as dist

import torch.distributed.distributed_c10d as dist_c10d

logger = logging.getLogger(__name__)
comm_timeout: int = None

def get_group_mapping() -> Dict[Tuple[str, Tuple[int, ...]], dist.ProcessGroup]:
    """return the mapping of (backend, global_ranks) to process group"""
    pg_maps = dist_c10d._pg_map
    ranks_to_group_mapping = dict()
    for process_group, (backend, store) in pg_maps.items():
        ranks = dist_c10d._pg_group_ranks[process_group]
        global_ranks = tuple(sorted(ranks.keys()))
        ranks_to_group_mapping[(backend, global_ranks)] = process_group
    return ranks_to_group_mapping

def get_group_by_ranks(ranks: Sequence[int],
                       backend: str='nccl') -> Optional[dist.ProcessGroup]:
    """if (backend, ranks) has been initialized, 
       return the process group, otherwise return None"""
    if backend is None:
        default_pg = dist_c10d._get_default_group()
        backend = dist_c10d._pg_map[default_pg][0]
    else:
        backend = dist.Backend(backend)
    ranks_to_group = get_group_mapping()
    ranks = tuple(sorted(ranks))
    return ranks_to_group.get((backend, ranks), None)

def _store_based_barrier(rank: int, store, timeout: int, world_size: int) -> None:
    """
    Barrier based on store which is used for synchronizing processes after
    ``init_process_group`` or ``new_group``. Intended to be used only with
    those two methods and is not a generic alternative to ``barrier()``.
    """
    store_key = "{}:{}".format(STORE_BASED_BARRIER_PREFIX, distributed_c10d._group_count)
    logger.info("Added key: {} to store for rank: {}, host_name: {}".format(store_key, rank, socket.gethostname()))
    store.add(store_key, 1)
    # time.sleep(0.05) # NOTE

    # Now wait for all workers to check in with the store.
    # Use 'add' instead of 'get' since for some store implementations 'add'
    # doesn't work well with 'get'. Ideally the store implementations should
    # be fixed, but for backward compatiblity reasons it is risky to change
    # the store implementations. Once, we completely migrate away from these
    # legacy stores, we can use 'get' here instead.
    worker_count = store.add(store_key, 0)
    start = time.time()
    log_time = time.time()
    while worker_count != world_size:
        time.sleep(0.01)
        # time.sleep(0.02) # NOTE
        worker_count = store.add(store_key, 0)

        # Print status periodically to keep track.
        if timedelta(seconds=(time.time() - log_time)) > timedelta(seconds=10):
            logger.info(
                "Waiting in store based barrier to initialize process group for "
                "rank: {}, key: {} (world_size={}, worker_count={}, timeout={},)".format(
                    rank, store_key, world_size, worker_count, timeout
                )
            )
            log_time = time.time()

        if timedelta(seconds=(time.time() - start)) > timeout:
            raise RuntimeError(
                "Timed out initializing process group in store based barrier on "
                "rank: {}, for key: {} (world_size={}, worker_count={}, timeout={})".format(
                    rank, store_key, world_size, worker_count, timeout
                )
            )

    logger.info(
        f"Rank {rank}: Completed store-based barrier for key:{store_key} with {world_size} nodes."
    )


def hulk_dist_new_group(ranks: Sequence[int],
                        timeout: timedelta=default_pg_timeout,
                        backend: Union[str, Backend]=None,
                        pg_options=None):
    """
    This function creates a new process group like `torch.distributed.new_group`,
    but will be synchronized only when current worker in group and group size > 1.
    """
    group = get_group_by_ranks(ranks, backend=backend)
    if group is not None:
        distributed_c10d._group_count += 1
        return group

    if backend == "gloo" and comm_timeout is not None:
        timeout = timedelta(seconds=comm_timeout)

    default_pg = _get_default_group()
    default_backend, default_store = distributed_c10d._pg_map[default_pg]
    global_rank = default_pg.rank()
    global_world_size = default_pg.size()

    # when current worker in group and group size > 1, we should barrier
    need_barrier: bool = True
    if global_rank not in ranks or len(ranks) == 1:
        need_barrier = False

    logger.debug(
        '=> [{}]/[{}] new group ranks={}, need_barrier={}, c10d._group_count={}.'.format(
            global_rank, global_world_size, ranks, need_barrier, distributed_c10d._group_count
        )
    )

    # Default to the same backend as the global process group
    # if the backend is not specified.
    if not backend:
        backend = default_backend

    # checks the input ranks
    assert ranks is not None, f"ranks is None is not allowed!"
    if ranks is not None:
        ranks = sorted(ranks)
        group_world_size = len(ranks)
        if group_world_size > global_world_size:
            raise RuntimeError(
                "the new group's world size should be less or "
                "equal to the world size set by "
                "init_process_group"
            )
        # check ranks' sanity
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise RuntimeError(
                    "The new group's rank should be within the "
                    "the world_size set by init_process_group"
                )
        if global_rank in ranks:
            group_rank = ranks.index(global_rank)
        else:
            group_rank = None
    else:
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank

    backend = Backend(backend)
    pg = _new_process_group_helper(
        group_world_size,
        group_rank,
        ranks,
        backend,
        default_store,
        pg_options=pg_options,
        timeout=timeout,
    )

    # Create the global rank to group rank mapping
    distributed_c10d._pg_group_ranks[pg] = {
        global_rank: group_rank for group_rank, global_rank in enumerate(ranks)
    }

    # barrier at the end to ensure that once we return from this method, all
    # process groups including global variables are updated correctly on all
    # ranks.
    if backend == Backend.MPI:
        # MPI doesn't have store.
        barrier()
    else:
        # Use store based barrier here since barrier() used a bunch of
        # default devices and messes up NCCL internal state.
        if need_barrier:
            logger.info("rank: {}, hostname:{}, start store_based_barrier".format(rank, socket.gethostname()))
            _store_based_barrier(global_rank, default_store, timeout, len(ranks))

        if default_backend == "hccl":
            if pg != GroupMember.NON_GROUP_MEMBER and get_backend(pg) in [
                Backend.GLOO,
                Backend.NCCL,
                Backend.HCCL,
            ]:
                pg._set_sequence_number_for_group()
        else:
            if pg != GroupMember.NON_GROUP_MEMBER and get_backend(pg) in [
                Backend.GLOO,
                Backend.NCCL,
            ]:
                pg._set_sequence_number_for_group()

    return pg

class ProcessGroupInitializer():
    """An object, knowing the parallelism configuration, that initializes parallel groups.

    Args:
        rank (int): The rank of current process.
        world_size (int): Size of whole communication world.
        config (Config): Running configuration.
        data_parallel_size (int): Size of data parallel.
        pipeline_parallel_size (int): Size of pipeline parallel.
        tensor_parallel_size (int): Size of tensor parallel.
    """
    isolated_group: Dict = None

    def __init__(self, 
                 rank: int,
                 world_size: int,
                 config: DictConfig,
                 data_parallel_size: int,
                 sequence_parallel_size: int,
                 pipeline_parallel_size: int,
                 tensor_parallel_size: int,
                 gloo_group_enabled: bool=True):

        self.config: DictConfig = config
        self.rank:int = rank
        self.world_size:int = world_size
        self.data_parallel_size:int = data_parallel_size
        self.sequence_parallel_size:int = sequence_parallel_size
        self.pipeline_parallel_size:int = pipeline_parallel_size
        self.tensor_parallel_size:int = tensor_parallel_size
        self.gloo_group_enabled: bool = gloo_group_enabled
        self.num_tensor_parallel_group = self.world_size // self.tensor_parallel_size
        super().__init__()

    # def init_dist_group(self):
    #     """Initialize data parallel groups, and assign local_ranks and groups to each gpu.

    #     Returns:
    #         Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
    #             A Data parallelism's information tuple.
    #     """
    #     dist_settings = list()
    #     num_pipeline_parallel_groups = self.world_size // self.pipeline_parallel_size
    #     for i in range(self.pipeline_parallel_size):
    #         start_rank = i * num_pipeline_parallel_groups
    #         end_rank = (i + 1) * num_pipeline_parallel_groups
    #         for j in range(self.tensor_parallel_size):
    #             dp_x_sp_ranks = list(range(start_rank+j, end_rank, self.tensor_parallel_size))
    #             group = hulk_dist_new_group(dp_x_sp_ranks)
    #             group_cpu = None
    #             if self.gloo_group_enabled:
    #                 group_cpu = hulk_dist_new_group(dp_x_sp_ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

    #             if self.rank in dp_x_sp_ranks:
    #                 dist_settings.append(
    #                     (
    #                         dp_x_sp_ranks.index(self.rank), len(dp_x_sp_ranks), group,
    #                         group_cpu, dp_x_sp_ranks, ParallelMode.DATA_X_SEQ
    #                     )
    #                 )
    #             sp_size = self.sequence_parallel_size
    #             num_sub_dp_group = sp_size
    #             num_sub_sp_group = len(dp_x_sp_ranks) // sp_size
    #             for m in range(num_sub_dp_group):
    #                 dp_ranks = dp_x_sp_ranks[m::sp_size]
    #                 dp_group = hulk_dist_new_group(dp_ranks)
    #                 dp_group_cpu = None
    #                 if self.gloo_group_enabled:
    #                     dp_group_cpu = hulk_dist_new_group(dp_ranks, backend='gloo') if dist.get_backend() != 'gloo' else dp_group
    #                 if self.rank in dp_ranks:
    #                     dist_settings.append(
    #                         (
    #                             dp_ranks.index(self.rank), len(dp_ranks), dp_group,
    #                             dp_group_cpu, dp_ranks, ParallelMode.DATA
    #                         )
    #                     )
    #             if sp_size > 1:
    #                 for n in range(num_sub_sp_group):
    #                     sp_ranks = dp_x_sp_ranks[n*sp_size: (n+1)*sp_size]
    #                     sp_group = hulk_dist_new_group(sp_ranks)
    #                     sp_group_cpu = None
    #                     if self.gloo_group_enabled:
    #                         sp_group_cpu = hulk_dist_new_group(sp_ranks, backend='gloo') if dist.get_backend() != 'gloo' else sp_group
    #                     if self.rank in sp_ranks:
    #                         dist_settings.append(
    #                             (
    #                                 sp_ranks.index(self.rank), len(sp_ranks), sp_group,
    #                                 sp_group_cpu, sp_ranks, ParallelMode.SEQ
    #                             )
    #                         )

    #     return dist_settings

    def init_dist_group(self):
        """Initialize tensor parallel groups, and assign local_ranks and groups to each gpu.

        Returns:
            Tuple (local_rank, group_world_size, process_group, ranks_in_group, mode):
                A Tensor parallelism's information tuple.
        """

        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR

        for i in range(self.num_tensor_parallel_group):
            ranks = list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))
            group = hulk_dist_new_group(ranks)
            group_cpu = None
            if self.gloo_group_enabled:
                group_cpu = hulk_dist_new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, cpu_group, ranks_in_group, mode
    @classmethod
    def build_dist_initializer(cls,
                               config: DictConfig,
                               rank: int,
                               world_size: int,
                               data_parallel_size: int,
                               sequence_parallel_size: int,
                               pipeline_parallel_size: int,
                               tensor_parallel_size: int,
                               *extra_args, **extra_kwargs):

        return cls(rank, world_size, config,
                   data_parallel_size,
                   sequence_parallel_size,
                   pipeline_parallel_size,
                   tensor_parallel_size,
                   *extra_args, **extra_kwargs)


def build_dist_initializer(name: str,
                           cfg: DictConfig,
                           rank: int,
                           world_size: int,
                           data_parallel_size: int,
                           sequence_parallel_size: int, 
                           pipeline_parallel_size: int,
                           tensor_parallel_size: int, 
                           *extra_args, **extra_kwargs) -> ProcessGroupInitializer:

    return ProcessGroupInitializer(rank,
                                   world_size, cfg,
                                   data_parallel_size,
                                   sequence_parallel_size,
                                   pipeline_parallel_size,
                                   tensor_parallel_size,
                                   *extra_args, **extra_kwargs)