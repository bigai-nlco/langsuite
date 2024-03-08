
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from overrides import EnforceOverrides
from plotly.graph_objects import Figure
from langsuite.suit import StructuredException
from langsuite.utils import logging
from langsuite.utils.registry import Registry

class Action(ABC):
    def make_info(self):
        info = dict()
        info["action"] = type(self)
        for k, v in self.__dict__.items():
            if not k in info:
                info[k] = v
        return info

    @abstractmethod
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        pass

    def _executable_assert(self) -> None:
        pass

    def exec(self) -> Tuple[bool, Dict[str, object]]:
        try:
            self._executable_assert()
            (success, info) = self._exec()
            info.update(self.make_info())
        except StructuredException as e:
            e.param_dict.update(self.make_info())
            logging.logger.warn("%s: %s", type(e).__name__, e.param_dict)
            raise e
        return (success, info)


class World(ABC, EnforceOverrides):
    def __init__(self, name):
        self.name = name
        # FIXME
        self.action_reg = Registry(name=f"World_{name}.ACTION")

    @property
    @abstractmethod
    def agents(self) -> dict:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def get_observation(self, agent_name: str) -> dict:
        pass

    @abstractmethod
    def step(
        self, agent_name: str, action_dict: dict
    ) -> Tuple[bool, Dict[str, object]]:
        pass

    @abstractmethod
    def render(self) -> Figure:
        pass