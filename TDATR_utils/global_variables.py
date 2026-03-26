from typing import Optional
from enum import Enum

from enum import Enum, EnumMeta
from typing import List
class TensorParallelEnv(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, *args, **kwargs):
        self.load(*args, **kwargs)

    def load(self,
             mode: Optional[str] = None,
             vocab_parallel: bool = False,
             parallel_input_1d: bool = False,
             summa_dim: int = None,
             tesseract_dim: int = None,
             tesseract_dep: int = None,
             depth_3d: int = None,
             input_group_3d=None,
             weight_group_3d=None,
             output_group_3d=None):
        self.mode = mode
        self.vocab_parallel = vocab_parallel
        self.parallel_input_1d = parallel_input_1d
        self.summa_dim = summa_dim
        self.tesseract_dim = tesseract_dim
        self.tesseract_dep = tesseract_dep
        self.depth_3d = depth_3d
        self.input_group_3d = input_group_3d
        self.weight_group_3d = weight_group_3d
        self.output_group_3d = output_group_3d

    def save(self):
        return dict(mode=self.mode,
                    vocab_parallel=self.vocab_parallel,
                    parallel_input_1d=self.parallel_input_1d,
                    summa_dim=self.summa_dim,
                    tesseract_dim=self.tesseract_dim,
                    tesseract_dep=self.tesseract_dep,
                    depth_3d=self.depth_3d,
                    input_group_3d=self.input_group_3d,
                    weight_group_3d=self.weight_group_3d,
                    output_group_3d=self.output_group_3d)


tensor_parallel_env = TensorParallelEnv()

# parallel modes
class ParallelMode(Enum):
    """This is an enumeration class containing all possible parallel modes.
    """

    GLOBAL = 'global'

    # common parallel
    DATA = 'data'

    # model parallel - containing tensor and pipeline parallel groups
    # this is added to facilitate amp and grad clipping in hybrid parallel
    MODEL = 'model'

    # pipeline parallel
    PIPELINE = 'pipe'

    # pipeline-head-tail
    PIPELINE_HEAD_TAIL = 'pipe_head_tail'

    # containing all ranks in tensor parallel
    TENSOR = 'tensor'

    # 1D Parallel
    PARALLEL_1D = '1d'

    # SP Parallel
    SEQ = 'SEQUENCE'

    # mixture group of data parallel and sp parallel
    DATA_X_SEQ = 'DATA_SEQUENCE'

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


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