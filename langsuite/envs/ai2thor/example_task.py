# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from langsuite.constants import WORKSPACE_PATH
from langsuite.envs.ai2thor.ai2thor_env import AI2ThorEnv
from langsuite.task import TASK_REGISTRY, BaseTask, TaskRunner
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder

__all__ = ["ExampleTask", "ExampleTaskRunner"]

AI2THORPath = Path(__file__).parent


def load_data(data_dir):
    try:
        out: Dict[str, Any] = {}
        os.chdir(Path(data_dir, "data", "procthor-10k"))
        exec(open(Path(data_dir, "data", "procthor-10k", "main.py")).read(), out)
        procthor_data = out["load_dataset"]()
        os.chdir(WORKSPACE_PATH)
    except Exception:
        import prior

        procthor_data = prior.load_dataset("procthor-10k")

    # create mock task data
    task_data = []
    for _id, world_data in enumerate(procthor_data["train"]):
        task_data.append(
            dict(
                name=f"ExampleTask:Procthor2DEnv:{_id}",
                data=dict(world_data=world_data),
                task_definition="",
                inputs=[],
                targets=[],
            )
        )
        if _id > 10:
            break
    # print(task_data[0])
    return task_data


@TASK_REGISTRY.register(name="ExampleTask:Procthor2DEnv")
class ExampleTask(BaseTask):
    def __init__(self, *, env, template, name, **kwargs) -> None:
        super().__init__(env=env, template=template, name=name, **kwargs)

        self.stop_criterions = [lambda _: self._timesteps >= 10]

        self._success_criteria = []

    @classmethod
    def create(cls, task_cfg, task_data=None):
        if not task_data:
            task_data = random.choice(load_data(AI2THORPath))

        env = AI2ThorEnv.create(task_cfg["env"])
        world_confg = deepcopy(task_cfg["world"])
        if "world_data" in task_data.get("data"):
            world_confg.update({"data": task_data["data"]["world_data"]})

        env.create_world(world_confg)
        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))

        for agent in task_cfg["agents"]:
            env.add_agent(agent)

        task = cls(
            env=env,
            template=task_cfg["template"],
            name=task_cfg.get("name", task_cfg["task"]),
        )
        return task

    def start(self, render=True):
        self.env.reset()
        if render:
            # broadcast to all agents
            logger.emit({"role": "system", "content": self.task_guidance})
        return self.step(action_dict={"prompt": self.task_guidance})

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

    def _determine_stop(self, cur_info):
        return any(stop_criterion(cur_info) for stop_criterion in self.stop_criterions)

    def run(self, render=True):
        obs, _, done, info = self.start()

        while not done:
            action_dict = dict()
            for aid, agnt_info in enumerate(info["n"]):
                agnt_name = self.env.agent_ids[aid]
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
                    action_dict[agnt_name] = {"prompt": agnt_info["feedback"]}
            obs, _, done, info = self.step(action_dict)

            if self._determine_stop(info):
                break
        if render:
            logger.emit({"role": "system", "content": "DONE!"})
            logger.emit("")
        return done


class ExampleTaskRunner(TaskRunner):
    def __init__(self, config) -> None:
        self.config = config

    def run_sample_task(self):
        task_data = random.choice(load_data(AI2THORPath))

        task = ExampleTask.create_from_config(self.config, task_data)

        task.run()

    def run(self):
        for task_sample in load_data(AI2THORPath):
            task = ExampleTask.create_from_config(self.config, task_sample)
            task.run()
