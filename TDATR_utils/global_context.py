#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import random
import socket
import operator
from itertools import chain
from functools import reduce
from datetime import timedelta
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Union, Dict

import torch
import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig
from TDATR_utils.constants import TENSOR_PARALLEL_MODES, INITIALIZER_MAPPING
from TDATR_utils.global_variables import tensor_parallel_env as env
from TDATR_utils.global_variables import ParallelMode
from TDATR_utils.global_variables import SingletonMeta
from TDATR_utils.process_group_initializer import build_dist_initializer

logger = logging.getLogger(__name__)


@dataclass
class Worker(object):
    hostname: str
    device_id: int
    global_rank: int
    data_parallel_rank: int
    sequence_paralle_rank: int
    data_sequence_parallel_rank: int
    model_parallel_rank: int
    pipeline_parallel_rank: int
    tensor_parallel_rank: int


class GlobalContext(metaclass=SingletonMeta):
    """This class provides interface functions for users to get the parallel context,
    such as the global rank, the local rank, the world size, etc. of each device.

    Note:
        The parallel_mode used in this class should be concluded in ``ParallelMode``.
        More details about ``ParallelMode`` could be found in
        `parallel_mode <https://github.com/hpcaitech/hulk/blob/main/hulk/context/parallel_mode.py>`_.
    """
    ParallelMode = ParallelMode


    def __init__(self):
        # distributed settings
        self._global_ranks = dict()
        self._local_ranks = dict()
        self._world_sizes = dict()
        self._groups = dict()
        self._cpu_groups = dict()
        self._monitor_groups = dict()
        self._ranks_in_group = dict()
        self._workers = dict()

        # global config
        self._config = None

        # default 3D parallel args, will be overwritten during process group intialization
        self.world_size = 1
        self.data_parallel_size = 1
        self.sequence_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.num_processes_on_current_node = -1
        self.virtual_pipeline_parallel_size = None
        self.virtual_pipeline_parallel_rank = None

        # logging
        self._verbose = False
        self._logger = logger

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose_: bool):
        self._verbose = verbose_

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_):
        if not isinstance(config_, DictConfig):
            raise TypeError("Invalid type for config, only DictConfig is supported")
        self._config = config_

    def detect_num_processes_on_current_node(self):
        hostname = socket.gethostname()
        hostname_list = [None for _ in range(self.get_world_size(ParallelMode.GLOBAL))]
        dist.all_gather_object(hostname_list, hostname, group=self.get_group(ParallelMode.GLOBAL))
        counter = Counter(hostname_list)
        self.num_processes_on_current_node = counter[hostname]

    def gather_workers(self) -> None:
        hostname = socket.gethostname()
        device_id = torch.cuda.current_device()
        worker = Worker(hostname=hostname,
                        device_id=device_id,
                        global_rank=self.get_global_rank(),
                        data_parallel_rank=self.get_local_rank(ParallelMode.DATA),
                        sequence_paralle_rank=self.get_local_rank(ParallelMode.SEQ),
                        data_sequence_parallel_rank=self.get_local_rank(ParallelMode.DATA_X_SEQ),
                        model_parallel_rank=self.get_local_rank(ParallelMode.MODEL),
                        tensor_parallel_rank=self.get_local_rank(ParallelMode.TENSOR),
                        pipeline_parallel_rank=self.get_local_rank(ParallelMode.PIPELINE))
        workers: List[Worker] = [None for _ in range(self.get_world_size(ParallelMode.GLOBAL))]
        dist.all_gather_object(workers, worker, group=self.get_group(ParallelMode.GLOBAL))
        self._workers = {worker.global_rank: worker for worker in workers}

    def get_worker_by_rank(self, global_rank: Optional[int]=None) -> Worker:
        if len(self._workers) == 0:
            raise ValueError(f"workers is empty, call `gpc.gather_workers()` first!")
        if global_rank is None:
            global_rank = self.get_global_rank()
        if global_rank not in self._workers:
            raise KeyError(f"global_rank={global_rank} does not exist in the workers.")
        return self._workers[global_rank]
    
    @staticmethod
    def _check_parallel_mode(parallel_mode: ParallelMode):
        assert isinstance(parallel_mode, ParallelMode), \
            f'expected the argument parallel_mode to be of enum ParallelMode, but got {type(parallel_mode)}'

    def get_global_rank(self):
        """Returns the global rank of the current device.

        Returns:
            int: The global rank of the current device
        """
        return self._global_ranks[ParallelMode.GLOBAL]

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int):
        """Adds the global rank of the current device for `parallel_mode` to the context.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The parallel mode for the rank.
            rank (int): The rank to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._global_ranks[parallel_mode] = rank

    def set_seed(self, seed: int):
        """Sets seeds for all random libraries.

        Args:
            seed (int): seed for random states
        """

        # Ensure that different pipeline MP states get different seed
        seed = seed + (100 * self._local_ranks.get(ParallelMode.PIPELINE, 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_local_rank(self, parallel_mode: ParallelMode):
        """Returns the local rank of the current device.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            int: The local rank of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        if not self.is_initialized(parallel_mode):
            return 0
        return self._local_ranks[parallel_mode]

    def init_global_dist(self, rank: int, world_size: int, backend: str, host: str, port: int):
        """Initializes the global distributed environment

        Args:
           rank (int): rank for the default process group.
           world_size (int): world size of the default process group.
           backend (str): backend for ``torch.distributed``
           host (str): the master address for distributed training.
           port (str): the master port for distributed training
        """
        # initialize the default process group
        init_method = f'tcp://{host}:{port}'
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method)
        # if self.config.common.ddp_comm_monitor_enable:
        #     _utils.comm_timeout = self.config.common.ddp_comm_monitor_timeout

        # None will give the default global process group for pytorch dist operations
        ranks = list(range(world_size))
        cpu_group = None
        self._register_dist(rank, world_size, dist.GroupMember.WORLD, cpu_group, ranks, ParallelMode.GLOBAL)
        self.add_global_rank(ParallelMode.GLOBAL, rank)
    
    def init_parallel_groups(self):
        """Initializes the parallel groups.

        Raises:
            AssertionError: Raises an AssertionError if the field parallel is not present in the config file.
        """

        # get rank and world size
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)
        self.world_size = world_size

        # set parallel size as attributes for global context
        self.sequence_parallel_size = self._config.model_parallel.sequence_parallel_size
        self.tensor_parallel_size = self._config.model_parallel.tensor_model_parallel_size
        self.pipeline_parallel_size = self._config.model_parallel.pipeline_model_parallel_size

        # the user should not set the data parallel size manually
        # instead, it should be calculated based on other parallel config
        mp_size = self.pipeline_parallel_size * self.tensor_parallel_size
        self.data_parallel_size = self.world_size // (self.sequence_parallel_size * mp_size)

        # get the tensor parallel mode and check
        tensor_parallel_mode = None
        # if self.tensor_parallel_size != 1:
        tensor_parallel_mode = TENSOR_PARALLEL_MODES(self._config.model_parallel.tensor_model_parallel_mode)
        assert tensor_parallel_mode in TENSOR_PARALLEL_MODES, \
            f"mode in the parallel config must be set to one of {TENSOR_PARALLEL_MODES}"
        env.mode = tensor_parallel_mode

        self.check_sanity()

        pg_init = []
        gloo_group_enabled = self._config.common.gloo_group_enabled
        # init data parallel process group for compatibility with other parallel module such as zero
        pg_init.append(dict(type=INITIALIZER_MAPPING['data'], gloo_group_enabled=gloo_group_enabled))

        # init model parallel process group for compatibility with amp and clip grad
        pg_init.append(dict(type=INITIALIZER_MAPPING['model'], gloo_group_enabled=gloo_group_enabled))

        # if self.pipeline_parallel_size > 1:
        #     pg_init.append(dict(type=INITIALIZER_MAPPING['pipeline']))
        pg_init.append(dict(type=INITIALIZER_MAPPING['pipeline'], gloo_group_enabled=gloo_group_enabled))
        pg_init.append(dict(type=INITIALIZER_MAPPING['tensor'], gloo_group_enabled=gloo_group_enabled))

        # init specific tensor parallel group
        if tensor_parallel_mode is not None:
            kwargs = {}
            if tensor_parallel_mode == "2.5d":
                kwargs = {
                        "depth": self._config.model_parallel.tensor_model_parallel_depth,
                        "gloo_group_enabled": gloo_group_enabled
                    }
            # add this config to initialize later
            mode_name = tensor_parallel_mode.value.lower()
            initializer_type = INITIALIZER_MAPPING[mode_name]
            pg_init.append(dict(type=initializer_type, **kwargs))

        # run initialization of different process groups
        # from .process_group_initializer._utils import assert_group_count
        for pg_cfg in pg_init:
            # assert_group_count()
            name = pg_cfg.pop('type')
            initializer = build_dist_initializer(name, self.config,
                                                 rank, world_size,
                                                 self.data_parallel_size,
                                                 self.sequence_parallel_size,
                                                 self.pipeline_parallel_size,
                                                 self.tensor_parallel_size,
                                                 **pg_cfg)
            parallel_setting = initializer.init_dist_group()
            if isinstance(parallel_setting, list):
                for args in parallel_setting:
                    self._register_dist(*args)
            else:
                self._register_dist(*parallel_setting)


    def _add_local_rank(self, parallel_mode: ParallelMode, rank: int):
        """Adds the local rank of the current device for `parallel_mode` to the context.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The parallel mode for the rank.
            rank (int): The rank to be added.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._local_ranks[parallel_mode] = rank

    def get_next_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the next device.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            int: The global rank of the next device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the previous device.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            int: The global rank of the previous device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank - 1) % world_size]

    def is_first_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the first one
        among its group for `parallel_mode`.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = self.get_local_rank(parallel_mode)
        return rank == 0

    def is_last_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the last one
        among its group for `parallel_mode`.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        return rank == world_size - 1

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self.virtual_pipeline_parallel_size is not None and self.virtual_pipeline_parallel_rank != 0:
                return False
        return self.is_first_rank(ParallelMode.PIPELINE)

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self.virtual_pipeline_parallel_size \
                    is not None and self.virtual_pipeline_parallel_rank != self.virtual_pipeline_parallel_size - 1:
                return False
        return self.is_last_rank(ParallelMode.PIPELINE)

    def get_world_size(self, parallel_mode: ParallelMode):
        """Returns the world size for `parallel_mode`.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            int: The world size for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        if not self.is_initialized(parallel_mode):
            return 1
        return self._world_sizes[parallel_mode]

    def _add_world_size(self, parallel_mode: ParallelMode, world_size: int):
        """Adds world size for `parallel_mode`.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The parallel mode correponding to the process group
            world_size (int): The world size to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._world_sizes[parallel_mode] = world_size

    def get_group(self, parallel_mode: ParallelMode):
        """Returns the group of the current device for `parallel_mode`.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            torch.distributed.ProcessGroup: The group of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._groups[parallel_mode]

    def _add_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup):
        """Adds the group of the current device for `parallel_mode`.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.
            group (torch.distributed.ProcessGroup): The group to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._groups[parallel_mode] = group

    def get_cpu_group(self, parallel_mode: ParallelMode):
        """Returns the Gloo group of the current device for `parallel_mode`.

        :param parallel_mode: The chosen parallel mode
        :type parallel_mode: :class:`hulk.core.ParallelMode`
        :raises AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
            of :class:`hulk.core.ParallelMode`
        :return: The group of the current device for `parallel_mode`
        :rtype: torch.distributed.ProcessGroup
        """
        self._check_parallel_mode(parallel_mode)
        if self._cpu_groups[parallel_mode] is None:
            raise RuntimeError("cpu group has not been initialized, try to set 'common.gloo_group_enabled=true'")
        return self._cpu_groups[parallel_mode]

    def get_monitor_group(self, ranks: Tuple[int, ...]) -> dist.ProcessGroupGloo:
        return self._monitor_groups[tuple(sorted(ranks))]
    
    def _add_cpu_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup):
        """Adds the Gloo group of the current device for `parallel_mode`.

        :param parallel_mode: The chosen parallel mode
        :type parallel_mode: :class:`hulk.core.ParallelMode`
        :param group: The group to be added
        :type group: torch.distributed.ProcessGroup
        :raises AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
            of :class:`hulk.core.ParallelMode`
        """
        self._check_parallel_mode(parallel_mode)
        self._cpu_groups[parallel_mode] = group

    def _add_monitor_group(self, ranks_in_group: Tuple[int]) -> None:
        sorted_ranks = tuple(sorted(ranks_in_group))
        if not self.config.common.ddp_comm_monitor_enable:
            self._monitor_groups[sorted_ranks] = None
            return
        monitor_group = None
        for mode, group in self._cpu_groups.items():
            ranks = tuple(sorted(self.get_ranks_in_group(mode)))
            if ranks == sorted_ranks:
                monitor_group = group
                break
        assert monitor_group is not None, "monitor group not found"
        self._monitor_groups[sorted_ranks] = monitor_group

    def get_ranks_in_group(self, parallel_mode: Optional[ParallelMode]=None,
                           group: Optional[dist.ProcessGroup]=None) -> Tuple[int, ...]:
        """Returns the rank of the current device for `parallel_mode` in the group.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.

        Returns:
            int: The rank of the current device for `parallel_mode` in the group.
        """
        if parallel_mode is None and group is None:
            raise ValueError("`parallel_mode` and `group` are None!")
        if group is not None:
            if group not in dist.distributed_c10d._pg_group_ranks:
                raise RuntimeError(f"Illegal group: {group}")
            global2local = dist.distributed_c10d._pg_group_ranks[group]
            return sorted(global2local.keys())
        self._check_parallel_mode(parallel_mode)
        return self._ranks_in_group[parallel_mode]

    def _add_ranks_in_group(self, parallel_mode: ParallelMode, ranks: list):
        """Adds the ranks of the current device for `parallel_mode` in the group.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.
            ranks (list): List of ranks to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`hulk.core.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._ranks_in_group[parallel_mode] = ranks

    
    def _register_dist(self, local_rank, world_size, process_group, cpu_group, ranks_in_group, mode):
        self._add_local_rank(mode, local_rank)
        self._add_world_size(mode, world_size)
        self._add_group(mode, process_group)
        self._add_cpu_group(mode, cpu_group)
        self._add_ranks_in_group(mode, ranks_in_group)
        self._add_monitor_group(ranks_in_group)

    def check_sanity(self):
        """Checks sanity of the parallel context.

        Raises:
            AssertionError: Raises an AssertionError if the world size does not equal to the product
                of data parallel size, pipeline parallel size and tensor parallel size.
        """
        dps = self.data_parallel_size
        sps = self.sequence_parallel_size
        pps = self.pipeline_parallel_size
        tps = self.tensor_parallel_size
        ws = self.world_size
        assert ws == dps *sps * pps * tps, \
            f"Expected the world size {ws} to be equal to data" \
            f" parallel size ({dps}) * seq parallel size ({sps}) * " \
            f"pipeline parallel size ({pps}) * tensor parallel size ({tps})"

    

    def is_initialized(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether `parallel_mode` is initialized
        in the current system.

        Args:
            parallel_mode (:class:`hulk.core.ParallelMode`): The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether `parallel_mode` is initialized in the current system.
        """
        return parallel_mode in self._groups

    def destroy(self):
        """Destroys the current distributed parallel environment.
        """
        for mode, group in self._groups.items():
            if mode is not ParallelMode.GLOBAL:
                dist.destroy_process_group(group)
        # destroy global process group
        dist.destroy_process_group()
        self._groups.clear()

    def set_device(self, device_ordinal: int = None):
        """Sets distributed processes to be bound to devices.

        Args:
           device_ordinal (int, optional): the device id to be bound to
        """
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = torch.cuda.device_count()
            device_ordinal = global_rank % devices_per_node

        torch.cuda.set_device(device_ordinal)
        if self._verbose:
            self._logger.info(f'process rank {global_rank} is bound to device {device_ordinal}')

    

    def set_virtual_pipeline_parallel_size(self, size):
        self.virtual_pipeline_parallel_size = size

    def set_virtual_pipeline_parallel_rank(self, rank):
        self.virtual_pipeline_parallel_rank = rank


global_context = GlobalContext()



class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name 
    are not used concurrently."""

    def __init__(self):
        self.buffer: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}

    def get_tensor(self,
                   tensor_shape: Union[Sequence[int], torch.Size],
                   dtype: torch.dtype,
                   name: str) -> torch.Tensor:
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or \
                self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = \
                torch.empty(required_len,
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False)

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)

    def release_all_memory(self) -> None:
        for k, v in self.buffer.items():
            v.data.resize_(0)
            v.data = None
        self.buffer.clear()


global_memory_buffer = GlobalMemoryBuffer()