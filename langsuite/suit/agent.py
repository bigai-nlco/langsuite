

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from matplotlib.dates import TU

from overrides import EnforceOverrides
from langsuite.cli.cmd_cli import CMDClient

from langsuite.suit import World

class LangSuiteAgent(ABC, EnforceOverrides):
    def __init__(self, name, world, agent_data, step_limit=100):
        self.name: str = name
        self.world: World = world
        self.step_limit = step_limit

        self._stopped = False
        self._ready_to_stop = False

        self.cmd_cli: Optional[CMDClient] = None
        self.web_ui = None


    @abstractmethod
    def init(self, task_description: str, extra_info: Dict[str, str]) -> str:
        pass

    def update_config(self, config: dict):
        if 'step_limit' in config:
            self.step_limit = config['step_limit']
        self.body.update_config(config)

    @property
    def body(self):
        return self.world.agents[self.name]

    def setup(self, agent_config, cmd_cli, web_ui):
        self.cmd_cli = cmd_cli
        self.web_ui = web_ui

    # XXX isn't it a waste of lines?
    @classmethod
    def create(cls, name, world, **kwargs):
        return cls(name, world, **kwargs)

    def make_decision(self, last_surface_obs) -> dict:
        """
        Get action depending on recent obs (e.g., get LLM output)
        """
        if self._stopped:
            return {}
        if self._ready_to_stop:
            self._stopped = True
        return self.fetch_response(last_surface_obs)

    @abstractmethod
    def fetch_response(self, prompt: str, role="system") -> dict:
        pass

    @abstractmethod
    def pack(self, semantic_fb_obs):
        """
        Pack semantic feedback and observation to agent observations (e.g., make LLM input)
        """

    def pre_stop(self):
        self._ready_to_stop = True

    @property
    def stopped(self):
        return self._stopped
