# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from copy import deepcopy
from math import floor
from pathlib import Path

from tqdm import tqdm

from langsuite.actions.base_action import ActionFeedback
from langsuite.constants import WORKSPACE_PATH
from langsuite.envs.cwah.cwah_env import Cwah2DEnv
from langsuite.task import TASK_REGISTRY, BaseTask, TaskRunner
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder

CwahPath = WORKSPACE_PATH


def load_data(data_dir, subset=False):
    cwah_path = Path(data_dir, "cwah_test", "test_env_set_help.pik")
    # print(cwah_path)
    import pickle

    with open(cwah_path, "rb") as f:
        cwah_data = pickle.load(f)
    _id = 0
    cwah_train_data = []
    for cwah_sample in tqdm(cwah_data):
        _id += 1
        if subset and _id > subset:
            break
        cwah_train_data.append(
            dict(
                name=f"CwahTask:Cwah2DEnv:{_id}",
                data=dict(world_data=cwah_sample),
                task_definition="cwah task",
                inputs=[],
                targets=[],
                path=cwah_path,
            )
        )

    # print(task_data[0])
    return cwah_train_data


@TASK_REGISTRY.register(name="CwahTask:Cwah2DEnv")
class CwahTask(BaseTask):
    def __init__(self, *, env, template, name, **kwargs) -> None:
        """Create a new Teach task."""
        super().__init__(env=env, template=template, name=name, **kwargs)
        self.actions_taken = []
        self.actions_taken_success = []
        self.agent_locs = []
        self._num_steps_taken = 0

        self.stop_criterions = [lambda _: self._timesteps >= 50]

        def success_criterion(_history, cur_info, elapsed_timesteps):
            satisfied, unsatisfied, statistics = self.env.agents[0].check_progress(
                self.env.full_graph, self.env.goal_spec[0]
            )

            return sum(unsatisfied.values()) == 0

        self._success_criteria = [success_criterion]

    @classmethod
    def create(cls, task_cfg, task_data=None):
        if not task_data:
            # task_data = random.choice(load_data(CwahPath))
            task_data = load_data(task_cfg["dataroot"])[3]
        world_config = deepcopy(task_cfg["world"])
        if "world_data" in task_data.get("data"):
            world_config.update({"data": task_data["data"]["world_data"]})
        world_data = world_config["data"]

        num_agents = task_cfg["num_agents"]
        task_goal = world_data["task_goal"]
        print("task_goal: ", task_goal)
        if "total_goal" in world_data.keys():
            print("total_goal: ", world_data["total_goal"])
        goal_class = world_data["goal_class"]
        task_name = world_data["task_name"]
        full_graph = world_data["init_graph"]

        env_config = task_cfg["env"]
        env_config.update({"num_agents": num_agents})
        env_config.update({"task_goal": task_goal})
        env_config.update({"goal_class": goal_class})
        env_config.update({"task_name": task_name})
        env_config.update({"full_graph": full_graph})
        env = Cwah2DEnv.create(env_config)
        env.create_world(world_config)
        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))

        for agent in task_cfg["agents"]:
            env.add_agent(agent)
        # alice = task_cfg["agents"][0]
        # bob = task_cfg["agents"][1]
        # alice["prompt_template"] = env.feedback_builder.build(
        #     "intro", "task_instruction"
        # )
        # alice["generator_prompt_template"] = env.feedback_builder.build(
        #     "intro", "message_instruction"
        # )
        # bob["prompt_template"] = env.feedback_builder.build("intro", "task_instruction")
        # bob["generator_prompt_template"] = env.feedback_builder.build(
        #     "intro", "message_instruction"
        # )

        # env.add_agent(alice)
        # env.add_agent(bob)

        rooms = world_config["data"]["init_rooms"]

        for i, agent in env.agents.items():
            print(rooms[i])
            agent.position = env.random_agent_position_in_room(rooms[i])

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
            for prompt in self.task_guidance.values():
                logger.emit({"role": "system", "content": prompt})
        return self.step(action_dict={"prompt": self.task_guidance})

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
        obs, reward, done, info = self.env.step(action_dict)
        self._timesteps += 1
        reward = self._compute_reward_hook(info)
        self._is_successful = self._determine_success_hook(info)

        # done = self.env.is_terminated or self._is_successful
        return obs, reward, done, info

    def _determine_stop(self, cur_info):
        for cur_agent_info in cur_info["n"]:
            if cur_agent_info is None:
                continue
            if "is_terminated" in cur_agent_info and cur_agent_info["is_terminated"]:
                return True

        return any(stop_criterion(cur_info) for stop_criterion in self.stop_criterions)

    def run(self, render=True):
        obs, _, done, info = self.start()

        while True:
            action_dict = {}
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
                            logger.robot_emit(resp, name=agent_name, action=resp_action)
                    else:
                        raise ValueError(
                            f"Unable to render assistant response: {agnt_info['response']}"
                        )

                if "feedback" in agnt_info:
                    if render:
                        logger.emit(
                            {"role": "system", "content": agnt_info["feedback"]}
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
                satisifed, unsatisfied, statistics = self.env.agents[0].check_progress(
                    self.env.full_graph, self.env.goal_spec[0]
                )
                print("satisifed: ", satisifed)
                print("unsatisifed: ", unsatisfied)
                goal_condition_success_rate = str(statistics)
                logger.emit(
                    {
                        "role": "system",
                        "content": f"satisifed: {satisifed}\nunsatisifed: {unsatisfied}",
                    }
                )
                logger.emit(
                    {
                        "role": "system",
                        "content": f"\nGoal-Condition Success Rate: {goal_condition_success_rate}",
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
            oppo_name = self.env.agents[1 - agent_id].agent_name
            task_guidance_dict[agent_name] = self.env.feedback_builder.build(
                "intro",
                agent_name=agent_name,
                oppo_name=oppo_name,
                max_view_steps=agent.max_view_distance / agent.step_size,
                degree=floor(agent.aov / 2),
                max_inventory=agent.inventory_capacity,
                max_manipulation_steps=agent.max_manipulate_distance / agent.step_size,
                example=self._feedback_builder.build("example"),
            )

        return task_guidance_dict


class CwahTaskRunner(TaskRunner):
    def __init__(self, config) -> None:
        self.config = config

    def run_sample_task(self):
        task = CwahTask.create_from_config(self.config, task_data=None)

        task.run()

    def run(self):
        for task_sample in load_data(CwahPath):
            task = CwahTask.create_from_config(self.config, task_sample)
            task.run()
