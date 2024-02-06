# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from copy import deepcopy
import json
from math import floor
import os
from pathlib import Path
from math import floor

from langsuite.actions.base_action import ActionFeedback
from langsuite.envs.babyai.babyai_env import BabyAIEnv
from langsuite.envs.babyai.levels import level_dict
from langsuite.task import TASK_REGISTRY, BaseTask
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder

__all__ = [
    "BabyAITask",
]


def load_data(seed_size, stage):
    data = []
    if stage == "test":
        for level in level_dict:
            if "Test" in level:
                for seed in range(seed_size):
                    data.append(
                        {
                            "type": "BabyAIWorld",
                            "id": level,
                            "seed": seed,
                        }
                    )
    elif stage == "train":
        for level in level_dict:
            if "Test" not in level:
                for seed in range(seed_size):
                    data.append(
                        {
                            "type": "BabyAIWorld",
                            "id": level,
                            "seed": seed,
                        }
                    )
    else:
        for level in level_dict:
            for seed in range(seed_size):
                data.append(
                    {
                        "type": "BabyAIWorld",
                        "id": level,
                        "seed": seed,
                    }
                )
    return data


@TASK_REGISTRY.register(name="BabyAITask:BabyAIEnv")
class BabyAITask(BaseTask):
    """
    BabyAI tasks
    """

    def __init__(self, *, env, template, name, **kwargs) -> None:
        super().__init__(env=env, template=template, name=name, **kwargs)
        self.stop_criterions = [lambda _: self._timesteps >= 50]
        self._success_criteria = []
        self.task_done_checker = kwargs.get("task_done_checker", None)

    @classmethod
    def create(cls, task_cfg, task_data=None):
        env = BabyAIEnv.create(task_cfg["env"])
        world_confg = deepcopy(task_cfg["world"])
        if task_data:
            world_confg.update(task_data)

        env.create_world(world_confg)
        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))
        # propagate agent config to agents
        agent_cfg = {
            "max_view_distance": env.world.agent_view_size,
            "position": env.world.agent_pos,
            "agent_dir": env.world.agent_dir,
        }

        for agent in task_cfg["agents"]:
            cfg = deepcopy(agent_cfg)
            cfg.update(agent)
            env.add_agent(cfg)

        task = cls(
            env=env,
            template=task_cfg["template"],
            name=task_cfg.get("name", task_cfg["task"]),
        )
        return task

    def task_guidance(self):
        return self._feedback_builder.build(
            "intro",
            max_view_steps=self.env.world.agent_view_size,
            side_steps=floor(self.env.world.agent_view_size / 2),
            example=self._feedback_builder.build("example"),
        )

    def start(self, render=True):
        self.env.reset()
        if render:
            # broadcast to all agents
            prompt = self.task_guidance()
            logger.emit({"role": "system", "content": prompt})
        return self.step(action_dict={"prompt": prompt})

    def step(self, action_dict):
        if type(action_dict) == dict:
            if len(action_dict) == 0:
                info = {
                    "state": ActionFeedback(
                        success=False,
                        feedback="No action passed in.",
                    ),
                    "is_terminated": True,
                }
                return None, 0, False, info

        obs, reward, done, info = self.env.step(action_dict)
        self._timesteps += 1
        if done:
            info.update({"is_terminated": True})
        return obs, reward, done, info

    def _determine_stop(self, cur_info):
        if "is_terminated" in cur_info and cur_info["is_terminated"]:
            return True
        else:
            return any(
                stop_criterion(cur_info) for stop_criterion in self.stop_criterions
            )

    def _determine_success(self, cur_info):
        if "is_successfull" in cur_info and cur_info["is_successfull"]:
            self._is_successful = True
            return True
        return any(
            success_criteria(cur_info) for success_criteria in self._success_criteria
        )

    def run(self, render=True):
        obs, _, done, info = self.start()
        while True:
            action_dict = dict()
            if info:
                agent_id = info["agent"]
                # agnt_info = info["state"]
                agnt_info = info
                agnt_name = self.env.agents[agent_id].name
                if render and "response" in agnt_info:
                    if type(agnt_info["response"]) == str:
                        logger.robot_emit(
                            agnt_info["response"], name=agnt_name, action="chat"
                        )
                    elif type(agnt_info["response"]) == list:
                        for resp_action, resp in agnt_info["response"]:
                            logger.robot_emit(resp, name=agnt_name, action=resp_action)
                    else:
                        raise ValueError(
                            f"Unable to render assistant response: {agnt_info['response']}"
                        )

                if "feedback" in agnt_info:
                    if render:
                        logger.emit(
                            {"role": "system", "content": agnt_info["feedback"]}
                        )
                    action_dict = {"prompt": agnt_info["feedback"]}

            if self._determine_stop(info):
                if self._determine_success(info):
                    if render:
                        logger.emit(
                            {
                                "role": "system",
                                "content": self._feedback_builder.build(
                                    "BabyAIStop", "success"
                                ),
                            }
                        )
                else:
                    if render:
                        logger.emit(
                            {
                                "role": "system",
                                "content": self._feedback_builder.build(
                                    "BabyAIStop", "failure"
                                ),
                            }
                        )
                break

            obs, _, done, info = self.step(action_dict)

        if render:
            logger.emit({"role": "system", "content": "DONE!"})
            logger.emit("")
        return done

    # def is_task_done(self, info):
    #     # self.bot = Bot(self.env.world)
    #     action = self.env.bot.replan()
    #     if action and action == self.env.bot.mission.actions.done:
    #         return True
    #     print(action)
    #     return False
