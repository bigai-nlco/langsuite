# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
from copy import deepcopy
from math import floor
from pathlib import Path

from tqdm import tqdm

from langsuite.actions import get_action
from langsuite.actions.base_action import ActionFeedback
from langsuite.constants import WORKSPACE_PATH
from langsuite.envs.teach.libs.teach.dataset.dataset import Definitions
from langsuite.envs.teach.teach_env import Teach2DEnv
from langsuite.shapes import Point2D
from langsuite.task import TASK_REGISTRY, BaseTask, TaskRunner
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder


def load_data(data_dir, stage, subset=10):
    """
    Load TEACh (TEACh: Task-driven Embodied Agents that Chat) data from a specified directory.

    Args:
        data_dir (str): The directory containing TEACh data files.
        stage (str): Classification of data.

    Returns:
        list: One of task data dictionaries.
    """
    teach_train_data = []
    teach_paths = Path(
        data_dir, "data", "teach", "teach_test", stage
    ).glob("*.json")
    _id = 0
    for teach_path in tqdm(teach_paths):
        _id += 1
        if subset and _id > subset:
            break
        with open(teach_path) as f:
            teach_sample = json.load(f)
        teach_train_data.append(
            dict(
                name=f"TeachTask:Teach2DEnv:{_id}",
                data=dict(world_data=teach_sample),
                task_definition="teach task",
                inputs=[],
                targets=[],
                path=teach_path,
            )
        )
    return teach_train_data[0]


@TASK_REGISTRY.register(name="TeachTask:Teach2DEnv")
class TeachTask(BaseTask):
    """TEACh task class

    This class provides functions to:
        - Load environment, agents, task.
    """

    def __init__(self, *, env, template, name, **kwargs) -> None:
        """Create a new Teach task."""
        super().__init__(env=env, template=template, name=name, **kwargs)
        self.actions_taken = []
        self.actions_taken_success = []
        self.agent_locs = []
        self._num_steps_taken = 0

        self.stop_criterions = [
            lambda _: self.env.expert_steps >= len(self.env.interactions),
            lambda _: self._timesteps >= (500 if self.isExpert else 50),
        ]
        self.open_progress_check = get_action(
            action_name="OpenProgressCheck", env=self.env, agent=self.env.agents[0]
        )

        def success_criterion(_history, cur_info, elapsed_timesteps):
            open_progress_check_result = self.open_progress_check.step()
            return open_progress_check_result.success

        self._success_criteria = [success_criterion]

    @classmethod
    def create(cls, task_cfg, task_data=None):
        if not task_data:
            # task_data = random.choice(load_data(TeachPath))
            task_data = load_data(WORKSPACE_PATH, "test")
        print(task_data["path"])
        env = Teach2DEnv.create(task_cfg["env"])
        world_config = deepcopy(task_cfg["world"])
        if "world_data" in task_data.get("data"):
            world_config.update({"data": task_data["data"]["world_data"]})
        env.create_world(world_config)
        if task_cfg.get("isExpert", False):
            task_cfg["template"] = task_cfg["template"].replace(
                "react", "expert")
        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))

        # for agent in task_cfg["agents"]:
        #     env.add_agent(agent)
        commander = task_cfg["agents"][0]
        follower = task_cfg["agents"][1]

        commander["task_description"] = world_config["data"]["tasks"][0]["desc"]
        follower[
            "task_description"
        ] = "You can use `chat` to ask commander for task information."
        agents_config = [commander, follower]
        agents_info = world_config["data"]["tasks"][0]["episodes"][0]["initial_state"][
            "agents"
        ]
        for i, agent_info in enumerate(agents_info):
            agents_config[i]["position"] = Point2D(
                agent_info["position"]["x"], agent_info["position"]["z"]
            )
            agents_config[i]["rotation"] = agent_info["rotation"]["y"]
            agents_config[i]["isExpert"] = task_cfg.get("isExpert", False)
            env.add_agent(agents_config[i])
        task = cls(
            env=env,
            template=task_cfg["template"],
            name=task_cfg.get("name", task_cfg["task"]),
        )
        task.isExpert = task_cfg.get("isExpert", False)
        interactions = []
        for interaction in task_data["data"]["world_data"]["tasks"][0]["episodes"][0][
            "interactions"
        ]:
            if interaction["success"] == 1:
                interactions.append(interaction)
        env.interactions = interactions

        game_task = world_config["data"]["tasks"][0]
        definitions = Definitions(version="2.0")
        task_to_check = deepcopy(
            definitions.map_tasks_name2info[game_task["task_name"]]
        )  # Copying is important if you're sharing a definitions object across calls
        task_to_check.task_params = game_task["task_params"]
        env.task_to_check = task_to_check
        return task

    def start(self, render=True):
        """Return task introduction at beginning"""
        self.env.reset()
        self.original_goal_condition_success_rate = (
            self.calculate_subgoal_success_rate()
        )
        if render:
            # broadcast to all agents
            for prompt in self.task_guidance.values():
                logger.emit({"role": "system", "content": prompt})
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
                info = {"n": [{
                    "state": ActionFeedback(
                        success=False,
                        feedback="No action passed in.",
                    ),
                    "is_terminated": True,
                }]}
                return None, 0, False, info

        if type(action_dict) == str or (
            type(action_dict) == dict
            and list(action_dict.keys())[0] not in self.env.agent_ids
        ):
            # broadcast action
            action_dict_copy = deepcopy(action_dict)
            action_dict = {}
            for agent_id in self.env.agent_ids:
                agent_name = self.env.agents[agent_id].agent_name
                single_agent_action_dict = {
                    agent_id: action_dict_copy["prompt"][agent_name]
                }
                action_dict.update(single_agent_action_dict)
        if self.isExpert:
            obs, reward, done, info = self.env.expert_step(action_dict)
        else:
            obs, reward, done, info = self.env.step(action_dict)
        self._timesteps += 1
        reward = self._compute_reward_hook(info)
        self._is_successful = self._determine_success_hook(info)

        # done = self.env.is_terminated or self._is_successful
        return obs, reward, done, info

    def _determine_stop(self, cur_info):
        """
        Determine if the agent should stop based on stop criteria.

        Args:
            cur_info: Current information or state for stop criterion evaluation.

        Returns:
            bool: True if any stop criterion is met, False otherwise.
        """
        for cur_agent_info in cur_info["n"]:
            if cur_agent_info is None:
                continue
            if "is_terminated" in cur_agent_info and cur_agent_info["is_terminated"]:
                return True

        return any(stop_criterion(cur_info) for stop_criterion in self.stop_criterions)

    def calculate_subgoal_success_rate(self):
        """calculate the subgoal success rate"""
        original_format_objects = []
        for o in self.env.world.id2object.values():
            original_format_objects.append(o.props)
        progress_check_output = self.env.task_to_check.check_episode_progress(
            original_format_objects
        )
        task_desc, success, subgoals, gc_total, gc_satisfied = [
            progress_check_output["description"],
            progress_check_output["success"],
            progress_check_output["subgoals"],
            progress_check_output["goal_conditions_total"],
            progress_check_output["goal_conditions_satisfied"],
        ]
        statistics = gc_satisfied / gc_total
        return statistics

    def run(self, render=True):
        """agent action loop"""
        obs, _, done, info = self.start()

        while True:
            for obj in self.env.world.id2object.values():
                if obj.id in self.env.world.controlled_objects:
                    for controlled_obj_id in self.env.world.controlled_objects[obj.id]:
                        controlled_obj = self.env.world.id2object[controlled_obj_id]
                        controlled_obj.props["isToggled"] = obj.props["isToggled"]
            for obj in self.env.world.id2object.values():
                if obj.props["objectType"] in [
                    "Sink",
                    "SinkBasin",
                    "Bathtub",
                    "BathtubBasin",
                ]:
                    children = obj.find_all_children()
                    for child in children:
                        if child.props["dirtyable"] is True:
                            child.props["isDirty"] = False
                        if child.props["canFillWithLiquid"] is True:
                            child.props["isFilledWithLiquid"] = True
                            child.props["simbotIsFilledWithWater"] = True

                if obj.props["objectType"] in ["Toaster", "StoveBurner", "Microwave"]:
                    if obj.props["isToggled"] is True:
                        children = obj.find_all_children()
                        for child in children:
                            if child.props["cookable"] is True:
                                child.props["isCooked"] = True
                                child.props["simbotIsCooked"] = True
                            if child.props["isFilledWithLiquid"] is True:
                                grand_children = child.find_all_children()
                                for grand_child in grand_children:
                                    if "simbotIsBoiled" in grand_child.props:
                                        grand_child.props["simbotIsBoiled"] = True

                if obj.props["objectType"] in ["CoffeeMachine"]:
                    if obj.props["isToggled"] is True:
                        children = obj.find_all_children()
                        for child in children:
                            if child.props["canFillWithLiquid"] is True:
                                child.props["isFilledWithLiquid"] = True
                                child.props["simbotIsFilledWithCoffee"] = True

            action_dict = {}
            print(info)
            for aid, agnt_info in enumerate(info["n"]):
                # agent_name = self.env.agent_names[aid]
                if agnt_info is None:
                    continue
                agent_name = self.env.agents[aid].agent_name
                if render and "response" in agnt_info:
                    if type(agnt_info["response"]) == str:
                        logger.robot_emit(
                            agnt_info["response"], name=agent_name, action="chat"
                        )
                    elif type(agnt_info["response"]) == list:
                        for resp_action, resp in agnt_info["response"]:
                            logger.robot_emit(
                                resp, name=agent_name, action=resp_action)
                    else:
                        raise ValueError(
                            f"Unable to render assistant response: {agnt_info['response']}"
                        )

                if "feedback" in agnt_info:
                    if render:
                        logger.emit(
                            {"role": "system",
                                "content": agnt_info["feedback"]}
                        )
                        # pass
                    action_dict[aid] = {"prompt": agnt_info["feedback"]}
            if self._determine_stop(info) or self._is_successful:
                break
            obs, _, done, info = self.step(action_dict)
        if render:
            if self._is_successful:
                logger.emit(
                    {
                        "role": "system",
                        "content": "[SUCCESS] You have completed the task. Congratulations!",
                    }
                )
                logger.emit(
                    {
                        "role": "system",
                        "content": f"timesteps: {self._timesteps}",
                    }
                )
            else:
                logger.emit(
                    {
                        "role": "system",
                        "content": "[FAILURE] You failed to complete the task.",
                    }
                )
                goal_condition_success_rate = self.calculate_subgoal_success_rate()
                logger.emit(
                    {
                        "role": "system",
                        "content": f"Original Goal-Condition Success Rate: {self.original_goal_condition_success_rate}",
                    }
                )
                logger.emit(
                    {
                        "role": "system",
                        "content": f"Goal-Condition Success Rate: {goal_condition_success_rate}",
                    }
                )
            logger.emit({"role": "system", "content": "DONE!"})
            logger.emit("")

        return done

    @property
    def task_guidance(self):
        task_guidance_dict = {}
        for agent_id in self.env.agent_ids:
            agent = self.env.agents[agent_id]
            agent_name = agent.agent_name
            task_guidance_dict[agent_name] = self.env.feedback_builder.build(
                "intro",
                "task_instruction_for_" + agent_name,
                max_view_steps=agent.max_view_distance / agent.step_size,
                degree=floor(agent.aov / 2),
                max_inventory=agent.inventory_capacity,
                max_manipulation_steps=agent.max_manipulate_distance / agent.step_size,
                example=self._feedback_builder.build("example", agent_name),
            )

        return task_guidance_dict


class TeachTaskRunner(TaskRunner):
    def __init__(self, config) -> None:
        self.config = config

    def run_sample_task(self):
        task = TeachTask.create_from_config(self.config, task_data=None)

        task.run()

    def run(self):
        for task_sample in load_data(TeachPath):
            task = TeachTask.create_from_config(self.config, task_sample)
            task.run()
