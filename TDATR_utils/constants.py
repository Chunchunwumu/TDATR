# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, EnumMeta
from typing import List


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))

    def __contains__(cls, member):
        if not isinstance(member, Enum):
            if member is None:
                member = 'none'
            if isinstance(member, str):
                return member in cls._member_map_
            raise TypeError(
                "unsupported operand type(s) for 'in': '%s' and '%s'" % (
                    type(member).__qualname__, cls.__class__.__qualname__))
        return isinstance(member, cls) and member._name_ in cls._member_map_


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


LOG_FORMAT_CHOICES = ChoiceEnum(["json", "none", "simple", "tqdm"])
DDP_BACKEND_CHOICES = ChoiceEnum(
    [
        "c10d",  # alias for pytorch_ddp
        "fully_sharded",  # FullyShardedDataParallel from fairscale
        "legacy_ddp",
        "no_c10d",  # alias for legacy_ddp
        "pytorch_ddp",
        "megatron_ddp",
    ]
)
DDP_COMM_HOOK_CHOICES = ChoiceEnum(["none", "fp16"])
DATASET_IMPL_CHOICES = ChoiceEnum(["raw", "lazy", "cached", "mmap", "fasta", "huffman"])
GENERATION_CONSTRAINTS_CHOICES = ChoiceEnum(["ordered", "unordered"])
GENERATION_DECODING_FORMAT_CHOICES = ChoiceEnum(
    ["unigram", "ensemble", "vote", "dp", "bs"]
)
ZERO_SHARDING_CHOICES = ChoiceEnum(["none", "os", "os_v2"])
PRINT_ALIGNMENT_CHOICES = ChoiceEnum(["hard", "soft"])
CLIP_GRAD_NORM_TYPE_CHOICES = ChoiceEnum(["l2", "inf"])
RECOMPUTE_GRANULARITY_CHOICES = ChoiceEnum(["none", "selective", "selective_mlp_stage1", "selective_mlp_stage2", "full"])

# parallelism modes
TENSOR_PARALLEL_MODES = ChoiceEnum(['none', '1d', '2d', '2.5d', '3d', 'sequence'])
TENSOR_PARALLEL_MODE = 'tensor_parallel_mode'

# initializer
INITIALIZER_MAPPING = {
    'data': 'Initializer_Data',
    'tensor': 'Initializer_Tensor',
    'pipeline': 'Initializer_Pipeline',
    'embedding': 'Initializer_Embedding',
    '1d': 'Initializer_1D',
    '2d': 'Initializer_2D',
    '2.5d': 'Initializer_2p5D',
    '3d': 'Initializer_3D',
    'sequence': 'Initializer_Sequence',
    'model': 'Initializer_Model',
    'moe': 'Initializer_Moe'
}

# 3D parallelism groups
INPUT_GROUP_3D = 'input_group_3d'
WEIGHT_GROUP_3D = 'weight_group_3d'
OUTPUT_GROUP_3D = 'output_group_3d'

# Attributes of tensor parallel parameters
IS_TENSOR_PARALLEL = 'is_tensor_parallel'
NUM_PARTITIONS = 'num_partitions'
TENSOR_PARALLEL_ATTRIBUTES = ChoiceEnum([IS_TENSOR_PARALLEL, NUM_PARTITIONS])

# Tensor placement policy
TENSOR_SHARD_STRATEGY = ChoiceEnum(['tensor', 'bucket'])
TENSOR_PLACEMENT_POLICY = ChoiceEnum(['cpu', 'cuda', 'auto'])

# checkpoint io mode
CKPT_IO_STRATEGY = ChoiceEnum(['default', 'master', 'greedy_balance', 'balance'])
SEQ_PARALLEL_ALGO = ChoiceEnum(['ulysses', 'local_atten', 'memory_efficient_local_atten'])

# quant
ACT_QUANT_STRATEGY = ChoiceEnum(['per_token', 'per_tensor_dynamic', 'per_tensor_static'])
WEIGHT_QUANT_STRATEGY = ChoiceEnum(['per_channel', 'per_tensor'])

