# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

from typing import Union

from TDATR_utils.dataclass import HulkDataclass
from TDATR_utils.utils import merge_with_parent

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

REGISTRIES = {}
SIMPLE_REGISTRIES = {}


def setup_registry(registry_name: str, base_class=None, default=None, required=False):
    assert registry_name.startswith("--")
    registry_name = registry_name[2:].replace("-", "_")

    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()
    DATACLASS_REGISTRY = {}

    # maintain a registry of all registries
    if registry_name in REGISTRIES:
        return  # registry already exists
    REGISTRIES[registry_name] = {
        "registry": REGISTRY,
        "default": default,
        "dataclass_registry": DATACLASS_REGISTRY,
    }

    def build_x(cfg: Union[DictConfig, str, Namespace], *extra_args, **extra_kwargs):
        if isinstance(cfg, DictConfig):
            choice = cfg._name
            if choice and choice in DATACLASS_REGISTRY:
                dc = DATACLASS_REGISTRY[choice]
                cfg = merge_with_parent(dc(), cfg)
        elif isinstance(cfg, str):
            choice = cfg
            if choice in DATACLASS_REGISTRY:
                cfg = DATACLASS_REGISTRY[choice]()
        else:
            choice = getattr(cfg, registry_name, None)
            if choice in DATACLASS_REGISTRY:
                cfg = DATACLASS_REGISTRY[choice].from_namespace(cfg)

        if choice is None:
            if required:
                raise ValueError("{} is required!".format(registry_name))
            return None

        cls = REGISTRY[choice]
        if hasattr(cls, "build_" + registry_name):
            builder = getattr(cls, "build_" + registry_name)
        else:
            builder = cls

        return builder(cfg, *extra_args, **extra_kwargs)

    def register_x(name, dataclass=None):
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError(
                    "Cannot register duplicate {} ({})".format(registry_name, name)
                )
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    "Cannot register {} with duplicate class name ({})".format(
                        registry_name, cls.__name__
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError(
                    "{} must extend {}".format(cls.__name__, base_class.__name__)
                )

            if dataclass is not None and not issubclass(dataclass, HulkDataclass):
                raise ValueError(
                    "Dataclass {} must extend HulkDataclass".format(dataclass)
                )

            cls.__dataclass = dataclass
            if cls.__dataclass is not None:
                DATACLASS_REGISTRY[name] = cls.__dataclass

                cs = ConfigStore.instance()
                node = dataclass()
                node._name = name
                cs.store(name=name, group=registry_name, node=node, provider="hulk")

            REGISTRY[name] = cls

            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY, DATACLASS_REGISTRY


def setup_simple_registry(registry_name: str, base_class=None, default=None):
    """
        this registry factory doesn't requires dataclass,
        and can build instance by pass name many times.
    """
    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()

    # maintain a registry of all registries
    if registry_name in SIMPLE_REGISTRIES:
        return  # registry already exists
    SIMPLE_REGISTRIES[registry_name] = {
        "registry": REGISTRY,
        "default": default,
    }

    def build_x(name: str, cfg: Union[DictConfig, str, Namespace], *extra_args, **extra_kwargs):
        if name not in REGISTRY:
            raise ValueError(f"{name} has not been registered!")

        cls = REGISTRY[name]
        if hasattr(cls, "build_" + registry_name):
            builder = getattr(cls, "build_" + registry_name)
        else:
            builder = cls

        return builder(cfg, *extra_args, **extra_kwargs)

    def register_x(name, dataclass=None):
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError(
                    "Cannot register duplicate {} ({})".format(registry_name, name)
                )
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    "Cannot register {} with duplicate class name ({})".format(
                        registry_name, cls.__name__
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError(
                    "{} must extend {}".format(cls.__name__, base_class.__name__)
                )

            if dataclass is not None and not issubclass(dataclass, HulkDataclass):
                raise ValueError(
                    "Dataclass {} must extend HulkDataclass".format(dataclass)
                )

            REGISTRY[name] = cls
            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY


from types import ModuleType
from typing import List


class Registry:
    """This is a registry class used to register classes and modules so that a universal 
    object builder can be enabled.

    Args:
        name (str): The name of the registry .
        third_party_library (list, optional):
            List of third party libraries which are used in the initialization of the register module.
    """

    def __init__(self, name: str, third_party_library: List[ModuleType] = None):
        self._name = name
        self._registry = dict()
        self._third_party_lib = third_party_library

    @property
    def name(self):
        return self._name

    def register_module(self, module_class):
        """Registers a module represented in `module_class`.

        Args:
            module_class (class): The module to be registered.
        Returns:
            class: The module to be registered, so as to use it normally if via importing.
        Raises:
            AssertionError: Raises an AssertionError if the module has already been registered before.
        """
        module_name = module_class.__name__
        assert module_name not in self._registry, f"{module_name} not found in {self.name}"
        self._registry[module_name] = module_class

        # return so as to use it normally if via importing
        return module_class

    def get_module(self, module_name: str):
        """Retrieves a module with name `module_name` and returns the module if it has 
        already been registered before.

        Args:
            module_name (str): The name of the module to be retrieved.
        Returns:
            :class:`object`: The retrieved module or None.
        Raises:
            NameError: Raises a NameError if the module to be retrieved has neither been
            registered directly nor as third party modules before.
        """
        if module_name in self._registry:
            return self._registry[module_name]
        elif self._third_party_lib is not None:
            for lib in self._third_party_lib:
                if hasattr(lib, module_name):
                    return getattr(lib, module_name)
            raise NameError(f'Module {module_name} not found in the registry {self.name}')

    def has(self, module_name: str):
        """Searches for a module with name `module_name` and returns a boolean value indicating
        whether the module has been registered directly or as third party modules before.

        Args:
            module_name (str): The name of the module to be searched for.
        Returns:
            bool: A boolean value indicating whether the module has been registered directly or
            as third party modules before.
        """
        found_flag = module_name in self._registry

        if self._third_party_lib:
            for lib in self._third_party_lib:
                if hasattr(lib, module_name):
                    found_flag = True
                    break

        return found_flag


OPHOOKS = Registry("ophooks")
GRADIENT_HANDLER = Registry("gradient_handler")
FP32MODELS = Registry("fp32models")
