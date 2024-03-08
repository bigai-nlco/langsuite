from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from git import Object

from overrides import EnforceOverrides
from traitlets import default
from langsuite.cli.cmd_cli import CMDClient

from langsuite.suit import LangSuiteEnv
from langsuite.suit import StructuredException
from langsuite.suit import Action
from langsuite.suit import TASK_REGISTRY
from langsuite.suit.message import MessageHandler
from langsuite.utils import logging


class TaskAction(ABC):
    name: str = "N/A"

class TaskActionWrapper(TaskAction, EnforceOverrides):

    @property
    @abstractmethod
    def wrapped_action(self) -> Action:
        pass

    @property
    @abstractmethod
    def status_map(self) -> Dict[type, str]:
        pass

    def exec(self) -> Tuple[bool, Dict[str, object]]:
        try:
            success, info = self.wrapped_action.exec()
            self._post_process(info)
            if success:
                info["status"] = "success.default"
            else:
                info["status"] = "failure.default"
            info["action"] = self.__class__.__name__
            return (success, info)
        except StructuredException as e:
            logging.logger.warn(
                f"Exception of type {type(e).__name__} happened during execution of {type(self).__name__}"
            )
            if e.__class__ in self.status_map:
                e.param_dict["status"] = self.status_map[e.__class__]
            else:
                e.param_dict["status"] = "failure.default"
            e.param_dict["action"] = self.__class__.__name__
            return (False, e.param_dict)

    def _post_process(self, info: dict):
        info["action"] = type(self)


class TaskStatus:
    pass


@dataclass
class LangsuiteTask(ABC, EnforceOverrides):
    env: LangSuiteEnv
    task_description: str
    cmd_cli: Optional[CMDClient]
    web_ui: Optional[Object]
    ACTIONS: ClassVar[List[TaskAction]]

    @abstractmethod
    def __init__(self, task_data, task_cfg) -> None:
        self.env: LangSuiteEnv = LangSuiteEnv(task_data)
        self.task_description: str = task_data["task_description"]
        self.task_type: str = task_data["task_type"]
        self.target_status = self.make_status(task_data)
        self.cfg = task_cfg

    # TODO define a type for it.
    @abstractmethod
    def make_status(self, task_data) -> TaskStatus:
        pass

    @abstractmethod
    def make_handler(self) -> MessageHandler:
        pass

    @classmethod
    def create(cls, task_cfg, task_data, cmd_cli=None, web_ui=None) -> "LangsuiteTask":
        cfg = deepcopy(task_cfg)
        task_data = cls._convert_task_data_format(task_cfg, task_data)

        task = cls(task_data, cfg)

        for i, ag_id in enumerate(task.env.agents):
            agent = task.env.agents[ag_id]
            task_cfg["agents"][i]["parser"] = task.make_handler()
            agent.setup(task_cfg["agents"][i], cmd_cli, web_ui)

        for f in task.ACTIONS:
            task.env.world.action_reg.register(f)

        return task

    @classmethod
    @abstractmethod
    def _convert_task_data_format(cls, task_cfg, raw_task_data) -> dict:
        pass

    def run(self):
        action_dict = {}
        final_reward_dict = {}
        for ag_id, agent in self.env.agents.items():
            action_dict[ag_id] = agent.init(self.task_description)
        running = True

        while running:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(
                action_dict
            )
            for k in reward_dict:
                final_reward_dict[k] = reward_dict[k]
            action_dict = info_dict
            running = not all(
                terminated_dict[k] or truncated_dict[k] for k in self.env.agents.keys()
            )
            logging.logger.debug("running: %s", running)
        
        return final_reward_dict

def make_task(task: Union[str, dict], *args, **kwargs) -> LangsuiteTask:
    """Make a task"""
    if type(task) == dict:
        task_cfg = deepcopy(task)
        task_name = task["task"]
    else:
        task_name = task
        task_cfg = {"task": task}

    if TASK_REGISTRY.hasRegistered(task_name):
        return TASK_REGISTRY.get(task_name).create(task_cfg, *args, **kwargs)

    raise ValueError(f"Task {task_name} is not defined.")
