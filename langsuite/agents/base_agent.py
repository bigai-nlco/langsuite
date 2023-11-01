# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import copy
from typing import Dict

from langsuite.utils import Counter
from langsuite.utils.registry import Registry

AGENT_REGISTRY = Registry("agent")


class Agent:
    counter = Counter("agent")

    def __init__(self, agent_id: str = "", agent_config: Dict = {}) -> None:
        # self.position = position
        Agent.counter.step()
        self.id = agent_id if len(agent_id) > 0 else str(Agent.counter)
        agent_config.update({"id": self.id})
        self.init_cfg = agent_config
        self.cfg = copy.deepcopy(agent_config)
        self.env = None

    @classmethod
    def create(cls, agent_cfg: Dict, position=None):
        agent_type = agent_cfg.pop("type")
        if agent_type is None:
            agent_type = "SimpleAgent"

        if AGENT_REGISTRY.hasRegistered(agent_type):
            return AGENT_REGISTRY.get(agent_type).create(agent_cfg)
        else:
            raise NotImplementedError(f"Agent type {agent_type} does not exist!")

    def set_env(self, env):
        self.env = env

    def get_config(self):
        return self.cfg

    def is_valid_action(self, action):
        if self.env:
            return self.env.is_valid_action(action)
        return True
