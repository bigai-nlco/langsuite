# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
import json
import random
import re
import sys
from copy import deepcopy
from math import floor
from pathlib import Path
import time
import numpy as np

from langsuite.actions.base_action import ActionFeedback
from langsuite.envs.babyai.babyai_env import BabyAIAction, BabyAIEnv, RegisteredEnvList
from langsuite.task import TASK_REGISTRY, BaseTask
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder

__all__ = [
    "DemoTask",
]

DemoPath = Path(__file__).parent.parent.parent

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return lines

@TASK_REGISTRY.register(name="DemoTask:BabyAIEnv")
class DemoTask(BaseTask):
    def __init__(self, *, env, template, name, **kwargs) -> None:
        super().__init__(env=env, template=template, name=name, **kwargs)
        self.dialogues = kwargs.get("dialogues", None)

    @classmethod
    def create(cls, task_cfg, task_data=None):
        env = BabyAIEnv.create(task_cfg["env"])
        world_confg = deepcopy(task_cfg["world"])
        if task_data:
            world_confg.update(task_data)

        expert_config = {
            "expert_type": "BabyAIWorld",
            "expert_id": world_confg.get("id"),
            "expert_seed": world_confg.get("seed"),
        }
        env.create_world(world_confg)
        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))
        for agent in task_cfg["agents"]:
            agent.update({"expert_config": expert_config})
            env.add_agent(agent)
        dialogues = load_data(task_cfg["dialogue_log_path"])
        task = cls(
            env=env,
            template=task_cfg["template"],
            name=task_cfg.get("name", task_cfg["task"]),
            dialogues=dialogues,

        )
        return task
    
    def run(self, render=True):
        agnt_name = "Alfred Agent"
        for utterance in self.dialogues:
            utterance = json.loads(utterance)
            if "role" in utterance and utterance["role"] == "system":
                time.sleep(1)
                logger.emit(
                            {"role": "system", "content": utterance["content"]}
                        )
                
            elif "role" in utterance and utterance["role"] == "assistant":
                time.sleep(2)
                if "action" in utterance:
                    logger.robot_emit(
                            utterance["content"], name=agnt_name, action=utterance["action"]
                        )
                else:
                    print("error1")
                    raise
            else:
                print("error")
                raise
        return True

    # def run(self, render=True):
    #     while True:
    #         action_dict = dict()
    #         if info:
    #             logger.info(info)
    #             # agnt_info = info["state"]
    #             agnt_info = info["n"][0]
    #             agent_id = agnt_info["agent"]
    #             agnt_name = self.env.agents[agent_id].name
    #             if render and "response" in agnt_info:
    #                 if type(agnt_info["response"]) == str:
    #                     logger.robot_emit(
    #                         agnt_info["response"], name=agnt_name, action="chat"
    #                     )
    #                 elif type(agnt_info["response"]) == list:
    #                     for resp_action, resp in agnt_info["response"]:
    #                         logger.robot_emit(resp, name=agnt_name, action=resp_action)
    #                 else:
    #                     raise ValueError(
    #                         f"Unable to render assistant response: {agnt_info['response']}"
    #                     )

    #             if agnt_info.get("feedback") is not None:
    #                 if render:
    #                     logger.emit(
    #                         {"role": "system", "content": agnt_info["feedback"]}
    #                     )
    #                 action_dict = {"prompt": agnt_info["feedback"]}
    #                 # if "fail" in agnt_info["feedback"]:
    #                 #     self.env.render()
    #                 #     sys.exit()

    #         if self._determine_stop(info):
    #             logger.emit(
    #                 {"role": "system", "content": str(self.conditioned_success)}
    #             )
    #             if self._is_successful:
    #                 if render:
    #                     logger.emit(
    #                         {
    #                             "role": "system",
    #                             "content": self._feedback_builder.build(
    #                                 "Stop", "success"
    #                             ),
    #                         }
    #                     )
    #             else:
    #                 if render:
    #                     logger.emit(
    #                         {
    #                             "role": "system",
    #                             "content": self._feedback_builder.build(
    #                                 "Stop", "failure"
    #                             ),
    #                         }
    #                     )
    #             break

    #         obs, _, done, info = self.step(action_dict)
    #         self.env.update_object_props_after_action()

    #     if render:
    #         logger.emit({"role": "system", "content": "DONE!"})
    #         logger.emit("")
    #     return done