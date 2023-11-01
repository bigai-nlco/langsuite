# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import copy
from collections import defaultdict
from copy import deepcopy
from typing import Union

import gymnasium as gym
import numpy as np

from langsuite.agents import Agent
from langsuite.utils.logging import logger
from langsuite.utils.registry import Registry
from langsuite.world import World

ENV_REGISTRY = Registry("env")


class LangSuiteEnv(gym.Env):
    """Base class for LangSuiteEnv environments."""

    @classmethod
    def create(cls, env_config, *args, **kwargs):
        return cls(env_config)

    def __init__(self, env_config):
        self.cfg = env_config
        self.agents = defaultdict()
        self.agent_ids = []
        self.world = None
        self.objects_in_view = []
        self._history = dict()
        self._action_history = []
        self.seed = None
        self.feedback_builder = None
        # self.is_terminated = False

    def create_world(self, world_cfg) -> None:
        self.world = World.create(world_cfg)

    def add_agent(self, agent_cfg) -> None:
        if "position" in agent_cfg:
            if isinstance(agent_cfg["position"], list):
                pass
            elif agent_cfg.get("position").lower() == "random":
                position = self.random_world_position()
                agent_cfg.update({"position": position})
        logger.info(agent_cfg)
        agent = Agent.create(copy.deepcopy(agent_cfg))
        agent.set_env(self)
        self.agents[agent.id] = agent
        self.agent_ids.append(agent.id)
        self._history[agent.id] = {"obs": [], "reward": [], "done": [], "info": []}

    def set_random_seed(self, seed):
        np.random.seed(seed)

    def random_world_position(self):
        return np.random.randint([0, 0], [10, 10]).tolist()

    def step(self, action_dict):
        """Run one timestep of the environment's dynamics. When end of episode is reached,
        you are responsible for calling `reset()` to reset this environment's state. Accepts
        an action and returns a tuple (observation, reward, done, info).
        """
        if type(action_dict) == str:
            # broadcast action to all agents
            return self.step({agent: action_dict for agent in self.agent_ids})

        self._action_history.append(copy.deepcopy(action_dict))

        info_n = dict(all={}, n=[])
        for agnt, action in action_dict.items():
            if agnt not in self.agent_ids:
                raise ValueError(f"Unknown agent name: {agnt}")

            obs, reward, success, info = self.step_single_agent(
                agent_id=agnt, action=action
            )

            if agnt not in self._history:
                self._history[agnt] = {"obs": [], "reward": [], "done": [], "info": []}

            self._history[agnt]["obs"].append(obs)
            self._history[agnt]["reward"].append(reward)
            self._history[agnt]["done"].append(success)
            self._history[agnt]["info"].append(info)

        obs_n = [agnt["obs"][-1] for _, agnt in self._history.items()]
        reward_n = [agnt["reward"][-1] for _, agnt in self._history.items()]
        done_n = [agnt["done"][-1] for _, agnt in self._history.items()]
        info_n["n"] = [agnt["info"][-1] for _, agnt in self._history.items()]

        info_n = self._summarize_info(info_n)
        return obs_n, reward_n, done_n, info_n

    def determine_success(self, info, task_spec):
        if "is_terminated" in info and info["is_terminated"]:
            return True
        return False

    def _summarize_info(self, info):
        info["all"] = {}
        if len(info["n"]) == 1:
            for k, v in info["n"][0].items():
                info[k] = deepcopy(v)
        return info

    def step_single_agent(self, *, agent_id, action):
        raise NotImplementedError

    def reset(self):
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""
        raise NotImplementedError

    def render(self, mode="", **kwargs):
        """Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        raise NotImplementedError

    def close(self):
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""
        raise NotImplementedError


def make_env(env: Union[str, dict], *args, **kwargs) -> LangSuiteEnv:
    """Create a environment"""
    if isinstance(env, str):
        env_type = env
        env_cfg = {"type": env}
    elif "type" in env:
        env_type = env["type"]
        env_cfg = copy.deepcopy(env)
    else:
        raise ValueError("Environment type must be provided")

    if ENV_REGISTRY.hasRegistered(env_type):
        return ENV_REGISTRY.get(env_type).create(env_cfg, *args, **kwargs)

    raise NotImplementedError(f"Environment {env_type} not found.")
