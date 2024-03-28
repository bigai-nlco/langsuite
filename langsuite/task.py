from __future__ import annotations

import copy
from copy import deepcopy
from typing import Union

import gymnasium as gym

import langsuite
from langsuite.envs.base_env import LangSuiteEnv
from langsuite.utils.logging import logger
from langsuite.utils.registry import Registry
from langsuite.utils.template_builder import TemplateBuilder

TASK_REGISTRY = Registry("task")


class BaseTask(gym.Wrapper):
    """
    Base class for all tasks.
    """

    def __init__(self, *, env, template, name, **kwargs) -> None:
        if not isinstance(env, LangSuiteEnv):
            env = langsuite.make_env(env)
        super().__init__(env=env)
        self.name = name
        self._is_successful: bool = False
        self._feedback_builder: str = TemplateBuilder(template_json=template)
        self._task_guidance = self._feedback_builder.build("intro")
        self._history = []
        self._success_criteria = []
        self._reward_fns = []
        self._pre_info_dict = None
        self._timesteps = 0

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(**kwargs)

    @property
    def is_successful(self) -> bool:
        return self._is_successful

    @property
    def task_guidance(self):
        return self._task_guidance

    def reset(self):
        obs = self.env.reset()
        self._history.clear()
        self._timesteps = 0
        self._pre_info_dict = copy.deepcopy(self.env.prev_info)
        return obs

    def step(self, action_dict):
        if type(action_dict) == dict:
            if len(action_dict) == 0:
                return None, 0, False, {"is_terminated": True}

        if type(action_dict) == str or (
            type(action_dict) == dict
            and list(action_dict.keys())[0] not in self.env.agent_ids
        ):
            # broadcast action
            action_dict = {agent: action_dict for agent in self.env.agents.keys()}

        obs, _, _, info = self.env.step(action_dict)
        self._timesteps += 1
        reward = self._compute_reward_hook(info)
        self._is_successful = self._determine_success_hook(info)

        done = self.env.is_terminated or self._is_successful
        return obs, reward, done, info

    def run(self, render=True):
        raise NotImplementedError

    def _compute_reward_hook(self, cur_info):
        return sum(
            [
                reward_fn(self._history, cur_info, timestamps=self._timesteps)
                for reward_fn in self._reward_fns
            ]
        )

    def _determine_success_hook(self, cur_info):
        return any(
            [
                check_success(
                    self._history, cur_info, elapsed_timesteps=self._timesteps
                )
                for check_success in self._success_criteria
            ]
        )

    def build_prompt(self, **kwargs):
        self._template_builder.build(**kwargs)


def make(task: Union[str, dict], *args, **kwargs):
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


class TaskRunner:
    def __init__(self, task) -> None:
        pass

    def metrics(self):
        raise NotImplementedError

    def run_task(self, task):
        logger.info(f"Working on task '{task.name}'")

    def run(self):
        for iter, task in enumerate(self.tasks):
            self.run_task(task)
