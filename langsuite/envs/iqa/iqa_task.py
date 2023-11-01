# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from pathlib import Path

from langsuite.envs.iqa import Iqa2DEnv
from langsuite.task import TASK_REGISTRY, BaseTask, TaskRunner
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder

__all__ = ["IqaTask"]

IqaPath = Path(__file__).parent.parent.parent.parent


def load_data(data_dir):
    """
    Load IQA (IQA: Visual Question Answering in Interactive Environments) data from a specified directory.

    Args:
        data_dir (str): The directory containing IQA data files.

    Returns:
        list: A list of task data dictionaries, each containing world and question-answer pairs.
    """
    iqa_data = json.load(open(Path(data_dir, "data", "iqa", "iqa_list_qa.json")))
    # iqa_data = json.load(open(Path(data_dir, "data", "iqa", "iqa_list_qa_counts_300.json")))

    task_data = []
    for _id, world_data in enumerate(iqa_data):
        task_data.append(
            dict(
                name=f"Iqa:Iqa2DEnv:{_id}",
                data=dict(world_data=world_data[0]),
                task_definition="",
                inputs=[],
                targets=[],
                qa=world_data[1],
            )
        )
    return task_data


def success_or_not(info, gold_answer="True"):
    """
    Check if the inferred answer matches the expected answer.

    Args:
        info: inferred answer to be checked.
        gold_answer (str): The expected answer. Default is "True".

    Returns:
        bool: True if the inferred answer matches the expected answer, False otherwise.
    """
    answer = extract_content(info)
    if answer is None:
        return False
    if str(answer).lower() == str(gold_answer).lower():
        return answer
    return False


@TASK_REGISTRY.register(name="IqaTask:Iqa2DEnv")
class IqaTask(BaseTask):
    """IQA task class

    This class provides functions to:
        - Load environment, agents, question-answer pair.
    """

    def __init__(self, *, env, template, name, **kwargs) -> None:
        super().__init__(env=env, template=template, name=name, **kwargs)
        self._is_successful: bool = False
        self.success_criterions = [success_or_not]
        self.stop_criterions = [lambda _: self._timesteps >= 100]

    @classmethod
    def create(cls, task_cfg, task_data=None):
        if not task_data:
            task_data = random.choice(load_data(IqaPath))

        env = Iqa2DEnv.create(task_cfg["env"])
        world_confg = deepcopy(task_cfg["world"])
        if "world_data" in task_data.get("data"):
            world_confg.update({"data": task_data["data"]["world_data"]})

        env.create_world(world_confg)
        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))
        env.question_type = task_cfg["question_type"]
        env.question = task_data["qa"][env.question_type]["question"]
        env.answer = task_data["qa"][env.question_type]["answer"]
        env.question_info["object_class"] = task_data["qa"][env.question_type][
            "object_class"
        ]
        if "recept" in task_data["qa"][env.question_type]:
            env.question_info["recept"] = task_data["qa"][env.question_type]["recept"]
        for agent in task_cfg["agents"]:
            env.add_agent(agent)

        task = cls(
            env=env,
            template=task_cfg["template"],
            name=task_cfg.get("name", task_cfg["task"]),
        )
        return task

    def start(self, render=True):
        """Return task introduction at beginning"""
        self.env.reset()
        if render:
            # broadcast to all agents
            for _, agent in self.env.agents.items():
                self._task_guidance = self._feedback_builder.build(
                    "intro",
                    degree=agent.view_degree,
                    max_manipulation_steps=agent.max_manipulate_distance,
                    max_view_steps=agent.max_view_distance,
                )
                logger.emit({"role": "system", "content": self.task_guidance})
        return self.step(action_dict={"prompt": self.task_guidance})

    def step(self, action_dict):
        """
        Perform a step in the environment based on given actions.

        Args:
            action_dict (dict or str): Actions to be taken by agents in the environment.

        Returns:
            tuple: Observation, reward, done flag, and additional information.
        """
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
        """
        Determine if the agent should stop based on stop criteria.

        Args:
            cur_info: Current information or state for stop criterion evaluation.

        Returns:
            bool: True if any stop criterion is met, False otherwise.
        """
        return any(stop_criterion(cur_info) for stop_criterion in self.stop_criterions)

    def _determine_success(self, cur_info, answer):
        return any(
            success_criterion(cur_info, answer)
            for success_criterion in self.success_criterions
        )

    def run(self, render=True):
        """agent action loop"""
        obs, _, done, info = self.start()

        while not done:
            action_dict = dict()
            if "n" not in info:
                break
            for aid, agnt_info in enumerate(info["n"]):
                if (
                    "response" in agnt_info
                    and "answer" in agnt_info["response"].lower()
                ):
                    self._is_successful = self._determine_success(
                        agnt_info["response"], self.env.answer
                    )
                agnt_name = self.env.agent_ids[aid]
                if render and "response" in agnt_info:
                    if type(agnt_info["response"]) == str:
                        logger.robot_emit(
                            agnt_info["response"], name=agnt_name, action="chat"
                        )
                        if "is_terminated" in info or (
                            "n" in info and "answer" in info["n"][0]["response"].lower()
                        ):
                            break
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
                if self.env.question_type == 2:
                    logger.robot_emit(
                        "answer [{}]".format(self.env.count_number),
                        name=agnt_name,
                        action="chat",
                    )
                    self._is_successful = self._determine_success(
                        "answer [" + str(self.env.count_number) + "]", self.env.answer
                    )
                else:
                    logger.robot_emit("answer [False]", name=agnt_name, action="chat")
                    self._is_successful = self._determine_success(
                        "answer [False]", self.env.answer
                    )
                break
        if render:
            if self._is_successful:
                if render:
                    logger.emit(
                        {
                            "role": "system",
                            "content": self._feedback_builder.build("Stop", "success"),
                        }
                    )
            else:
                if render:
                    logger.emit(
                        {
                            "role": "system",
                            "content": self._feedback_builder.build("Stop", "failure"),
                        }
                    )
            logger.emit({"role": "system", "content": "DONE!"})
            logger.emit("")
        return done


class ExampleTaskRunner(TaskRunner):
    def __init__(self, config) -> None:
        self.config = config

    def run_sample_task(self):
        task_data = random.choice(load_data(IqaPath))

        task = IqaTask.create_from_config(self.config, task_data)


def extract_content(input_string):
    """
    Extract answer enclosed in square brackets from the input string.

    Args:
        input_string (str): The input string containing content in square brackets.

    Returns:
        str or None: The extracted content, or None if no content is found.
    """
    pattern = r"{}\s*\[([^]]+)\]".format("Answer")
    match = re.search(pattern, input_string)
    if match:
        content = match.group(1)
        return content

    pattern = r"{}\s*\[([^]]+)\]".format("answer")
    match = re.search(pattern, input_string)
    if match:
        content = match.group(1)
        return content
    else:
        return None
