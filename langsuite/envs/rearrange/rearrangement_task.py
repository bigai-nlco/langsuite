# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import random
from copy import deepcopy
from math import floor
from pathlib import Path

import requests

from langsuite.actions.base_action import ActionFeedback
from langsuite.envs.rearrange.rearrange_env import Rearrange2DEnv
from langsuite.task import TASK_REGISTRY, BaseTask
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder

__all__ = [
    "RearrangementTask",
]

RearrangePath = Path(__file__).parent.parent.parent.parent

def load_data(data_dir, stage):
    def get_obj_id(world_data, obj_name):
        for object in world_data["objects"]:
            if object["name"] == obj_name:
                return object["objectId"]
        return None

    with open(
        Path(data_dir, "data", "rearrange", stage + ".json"),
#        Path(data_dir, "data", "rearrange", stage + ".json"),
        "r",
        encoding="utf-8",
    ) as scene_f:
        data = scene_f.readlines()
    # task_path = Path(data_dir, "resource", "rearrange", "task", "val.pkl.gz")
    # task_json = compress_pickle.load(path=task_path)
    tasks = []
    for data_item in data:
        task_json = json.loads(data_item)
        world_data = task_json["end_scene"]
        start_world_data = task_json["start_scene"]
        agent_data = start_world_data["agent"]
        starting_pose = task_json["task"]["starting_poses"]
        target_poses = task_json["task"]["target_poses"]
        openable_data = task_json["task"]["openable_data"]
        starts = []
        targets = []
        openness = {}
        for o in openable_data:
            openness[o["objectName"]] = {
                "start_openness": o["start_openness"],
                "target_openness": o["target_openness"],
            }
        for o in starting_pose:
            position = {
                "x": round(o["position"]["x"], 2),
                "y": round(o["position"]["z"], 2),
            }
            if o["name"] in openness:
                obj = {
                    "name": o["name"],
                    "object_id": get_obj_id(start_world_data, o["name"]),
                    "position": position,
                    "openness": openness[o["name"]]["start_openness"],
                }
            else:
                obj = {
                    "name": o["name"],
                    "object_id": get_obj_id(start_world_data, o["name"]),
                    "position": position,
                }
            starts.append(obj)
        for o in target_poses:
            position = {
                "x": round(o["position"]["x"], 2),
                "y": round(o["position"]["z"], 2),
            }
            if o["name"] in openness:
                obj = {
                    "name": o["name"],
                    "object_id": get_obj_id(start_world_data, o["name"]),
                    "position": position,
                    "openness": openness[o["name"]]["start_openness"],
                }
            else:
                obj = {
                    "name": o["name"],
                    "object_id": get_obj_id(start_world_data, o["name"]),
                    "position": position,
                }
            targets.append(obj)

        id = task_json["scene"] + "_" + str(task_json["index"])
        # if id == "FloorPlan307_44":
        tasks.append(
            dict(
                name=f"RearrangementTask:Rearrange2DEnv:{id}",
                data=dict(world_data=world_data, agent_data=agent_data),
                task_definition="",
                inputs=starts,
                targets=targets,
                start_world_data=start_world_data,
                scene=task_json["scene"],
                index=task_json["index"],
            )
        )
        # break
    return tasks


@TASK_REGISTRY.register(name="RearrangementTask:Rearrange2DEnv")
class RearrangementTask(BaseTask):
    """
    Rearrangement tasks
    """

    def __init__(self, *, env, template, name, **kwargs) -> None:
        super().__init__(env=env, template=template, name=name, **kwargs)
        self.stop_criterions = [lambda _: self._timesteps >= 100]
        self._success_criteria = [
            lambda curr_info: self.is_pose_conditions_met(curr_info)
        ]
        self.target_pos = kwargs.get("target_pos", None)
        self.start_pos = kwargs.get("start_pos", None)
        self.misplaced_value = 0.0
        self.fixed_value = 0.0
        self.diff_count = kwargs.get("diff_count", 0)

    @classmethod
    def create(cls, task_cfg, task_data=None):
        if not task_data:
            # task_data = random.choice(load_data(RearrangePath))
            task_data = random.choice(load_data(RearrangePath, "test"))
            # task_data = load_data(RearrangePath, "train")
            # index = random.randint(0, len(task_data) - 1)
            # task_data = task_data[1337]
            # print("index", index)
            # logger.info("Task index is " + str(index))

        env = Rearrange2DEnv.create(task_cfg["env"])

        #HACK XXX
        if 'log_file' in task_data:
            setattr(env, 'task_log_file_name', task_data['log_file'])


        world_confg = deepcopy(task_cfg["world"])
        if "world_data" in task_data.get("data"):
            world_confg.update({"data": task_data["data"]["world_data"]})
        env.create_world(world_confg)
        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))
        if "agent_data" in task_data.get("data"):
            agent_data = task_data["data"]["agent_data"]
            agent_position = [
                agent_data.get("position").get("x"),
                agent_data.get("position").get("z"),
            ]
            agent_rotation = agent_data.get("rotation").get("y")
            for agent in task_cfg["agents"]:
                agent.update(
                    {
                        "position": agent_position,
                        "rotation": agent_rotation,
                        "target_status": task_data["targets"],
                        "start_status": task_data["inputs"],
                    }
                )
                env.add_agent(agent)

        env.target_pose_description = env.get_room_description()
        target_pos = env.get_all_object_pos()
        if "start_world_data" in task_data:
            world_confg.update({"data": task_data["start_world_data"]})
        env.reset_world(world_confg)
        start_pos = env.get_all_object_pos()
        diff_count = env.get_diff_count(task_data["targets"], task_data["inputs"])

        task = cls(
            env=env,
            template=task_cfg["template"],
            name=task_cfg.get("name", task_cfg["task"]),
            target_pos=target_pos,
            start_pos=start_pos,
            diff_count=diff_count,
        )
        return task

    def task_guidance(self):
        agent_id = list(self.env.agents.keys())[0]
        agent = self.env.agents[agent_id]
        return self._feedback_builder.build(
            "intro",
            max_view_steps=agent.max_view_distance / agent.step_size,
            degree=floor(agent.aov / 2),
            max_inventory=agent.inventory_capacity,
            max_manipulation_steps=agent.max_manipulate_distance / agent.step_size,
            example=self._feedback_builder.build("example"),
        )

    def start(self, render=True):
        self.env.reset()
        if render:
            prompt = self.task_guidance()
            logger.emit({"role": "system", "content": prompt})
        return self.step(action_dict={"prompt": prompt})

    def step(self, action_dict):
        if type(action_dict) == dict:
            if len(action_dict) == 0:
                info = {
                    "state": ActionFeedback(
                        success=False,
                        feedback=f"No action passed in.",
                    ),
                    "is_terminated": True,
                }
                return None, 0, False, info

        # if type(action_dict) == str or (
        #     type(action_dict) == dict
        #     and list(action_dict.keys())[0] not in self.env.agent_names
        # ):
        #     # broadcast action
        #     action_dict = {agent: action_dict for agent in self.env.agents.keys()}

        obs, reward, done, info = self.env.step(action_dict)
        self._timesteps += 1
        self._is_successful = self._determine_success(info)
        return obs, reward, done, info

    def _determine_stop(self, cur_info):
        if "is_terminated" in cur_info and cur_info["is_terminated"]:
            return True
        else:
            return any(
                stop_criterion(cur_info) for stop_criterion in self.stop_criterions
            )

    def _determine_success(self, cur_info):
        return all([success_fn(cur_info) for success_fn in self._success_criteria])

    def run(self, render=True):
        obs, _, done, info = self.start()

        while not done:
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
                    # if "fail" in agnt_info["feedback"]:
                    #     self.env.render()
                    #     sys.exit()

            if self._determine_stop(info):
                logger.emit(
                    {
                        "role": "system",
                        "content": str(
                            {
                                "misplaced": self.misplaced_value,
                                "fixed": self.fixed_value,
                            }
                        ),
                    }
                )
                if self._is_successful:
                    if render:
                        logger.emit(
                            {
                                "role": "system",
                                "content": self._feedback_builder.build(
                                    "Stop", "success"
                                ),
                            }
                        )
                else:
                    if render:
                        logger.emit(
                            {
                                "role": "system",
                                "content": self._feedback_builder.build(
                                    "Stop", "failure"
                                ),
                            }
                        )
                break
            obs, _, done, info = self.step(action_dict)
        if render:
            logger.emit({"role": "system", "content": "DONE!"})
            logger.emit("")
        return done

    def is_pose_conditions_met(self, curr_info):
        # current_pos = self.env.get_all_object_pos()
        # diff = self.get_pos_diff(current_pos, self.target_pos)
        self.misplaced_value = self.misplaced()
        self.fixed_value = self.fixed()
        if self.misplaced_value == 0 and self.fixed_value == 1:
            return True
        else:
            return False

    def get_pos_diff(self, pos1, pos2, start=False):
        diff = {}
        rand_room_id = list(self.world.rooms.keys())[0]
        rand_room_poly = self.world.rooms[rand_room_id].geometry
        width_grid_size = (rand_room_poly.x_max - rand_room_poly.x_min) / 3
        length_grid_size = (rand_room_poly.y_max - rand_room_poly.y_min) / 3
        for pos1_obj_id, pos1_obj in pos1.items():
            position_compare = False
            open_compare = False
            if pos1_obj_id not in pos2:
                continue
            pos2_obj = pos2[pos1_obj_id]
            pos1_x = floor(
                (pos1_obj["obj_position"][0] - rand_room_poly.x_min) / width_grid_size
            )
            pos1_y = floor(
                (pos1_obj["obj_position"][1] - rand_room_poly.y_min) / length_grid_size
            )
            pos2_x = floor(
                (pos2_obj["obj_position"][0] - rand_room_poly.x_min) / width_grid_size
            )
            pos2_y = floor(
                (pos2_obj["obj_position"][1] - rand_room_poly.y_min) / length_grid_size
            )
            if not start:
                if pos1_x == pos2_x and pos1_y == pos2_y:
                    position_compare = True
            else:
                if (
                    pos1_obj["obj_position"][0] == pos2_obj["obj_position"][0]
                    and pos1_obj["obj_position"][1] == pos2_obj["obj_position"][1]
                ):
                    position_compare = True
            if "isOpen" not in pos2_obj:
                pos2_obj.update({"isOpen": "None"})

            if "isOpen" not in pos1_obj:
                pos1_obj.update({"isOpen": "None"})

            if (
                pos2_obj["openable"] == pos1_obj["openable"]
                and pos2_obj["isOpen"] == pos1_obj["isOpen"]
            ):
                open_compare = True
            if not position_compare or not open_compare:
                diff[pos1_obj_id] = pos1_obj
        return diff

    def get_pos_intersect(self, pos1, pos2):
        intersect = {}
        rand_room_id = list(self.world.rooms.keys())[0]
        rand_room_poly = self.world.rooms[rand_room_id].geometry
        width_grid_size = (rand_room_poly.x_max - rand_room_poly.x_min) / 3
        length_grid_size = (rand_room_poly.y_max - rand_room_poly.y_min) / 3
        for pos1_obj_id, pos1_obj in pos1.items():
            position_compare = False
            open_compare = False
            if pos1_obj_id not in pos2:
                continue
            pos2_obj = pos2[pos1_obj_id]
            pos1_x = floor(
                (pos1_obj["obj_position"][0] - rand_room_poly.x_min) / width_grid_size
            )
            pos1_y = floor(
                (pos1_obj["obj_position"][1] - rand_room_poly.y_min) / length_grid_size
            )
            pos2_x = floor(
                (pos2_obj["obj_position"][0] - rand_room_poly.x_min) / width_grid_size
            )
            pos2_y = floor(
                (pos2_obj["obj_position"][1] - rand_room_poly.y_min) / length_grid_size
            )

            if pos1_x == pos2_x and pos1_y == pos2_y:
                position_compare = True

            if "isOpen" not in pos2_obj:
                pos2_obj.update({"isOpen": "None"})

            if "isOpen" not in pos1_obj:
                pos1_obj.update({"isOpen": "None"})

            if (
                pos2_obj["openable"] == pos1_obj["openable"]
                and pos2_obj["isOpen"] == pos1_obj["isOpen"]
            ):
                open_compare = True

            if position_compare and open_compare:
                intersect[pos1_obj_id] = pos1_obj
        return intersect

    def misplaced(self):
        # could be larger than 1
        misplaced = 0.0
        current_pos = self.env.get_all_object_pos()
        # all_diff = self.get_pos_diff(self.start_pos, self.target_pos)
        changed_obj_pose = self.get_pos_diff(current_pos, self.start_pos, start=True)
        misplaced_obj_pose = self.get_pos_diff(changed_obj_pose, self.target_pos)

        if self.diff_count > 0:
            misplaced = round(len(misplaced_obj_pose) / self.diff_count, 2)
        return misplaced

    def fixed(self):
        fixed = 0.0
        current_pos = self.env.get_all_object_pos()
        # all_diff = self.get_pos_diff(self.start_pos, self.target_pos)
        changed_obj_pose = self.get_pos_diff(current_pos, self.start_pos, start=True)
        # print(changed_obj_pose)
        fixed_obj_pose = self.get_pos_intersect(changed_obj_pose, self.target_pos)
        # print(fixed_obj_pose)
        if self.diff_count > 0:
            fixed = round(len(fixed_obj_pose) / self.diff_count, 2)
        return fixed
