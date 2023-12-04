# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import re
from copy import deepcopy
from math import floor
from pathlib import Path

import numpy as np

from langsuite.actions.base_action import ActionFeedback
from langsuite.envs.alfred.alfred_env import Alfred2DEnv
from langsuite.task import TASK_REGISTRY, BaseTask
from langsuite.utils.logging import logger
from langsuite.utils.template_builder import TemplateBuilder

__all__ = [
    "AlfredTask",
]

AlfredPath = Path(__file__).parent.parent.parent.parent


def load_data(data_dir, stage):
    tasks = []
    dev_path = Path(data_dir, stage + ".json")
    with open(
        dev_path,
        "r",
        encoding="utf-8",
    ) as data_f:
        dev_data = json.load(data_f)
    # dev_data = {"a": [1] * 11}
    for task_type, task_paths in dev_data.items():
        for task_path in task_paths:
            with open(
                Path(data_dir).joinpath(task_path),
                "r",
                encoding="utf-8",
            ) as scene_f:
                task_json = json.load(scene_f)

                scene = task_json["scene"]
                agent_init = scene["init_action"]
                agent_data = {
                    "rotation": {"y": agent_init["rotation"]},
                    "position": {"x": agent_init["x"], "z": agent_init["z"]},
                }
                expert_actions = []
                for action in task_json["plan"]["low_actions"]:
                    api_action = action["api_action"]
                    expert_actions.append(api_action)

                turk_annotations = task_json["turk_annotations"]
                task_definition = turk_annotations["anns"][0]["task_desc"]
                target_status = task_json["pddl_params"]
                task_type = task_json["task_type"]
                tasks.append(
                    dict(
                        name=f"AlfredTask:Alfred2DEnv:{id}",
                        data=dict(world_data=scene, agent_data=agent_data),
                        task_description=task_definition,
                        expert_actions=expert_actions,
                        task_path=task_path,
                        target_status=target_status,
                        task_type=task_type,
                    )
                )
    return tasks


@TASK_REGISTRY.register(name="AlfredTask:Alfred2DEnv")
class AlfredTask(BaseTask):
    """
    Rearrangement tasks
    """

    def __init__(self, *, env, template, name, **kwargs) -> None:
        super().__init__(env=env, template=template, name=name, **kwargs)
        self._is_successful: bool = False
        self._feedback_builder: str = TemplateBuilder(template_json=template)
        self._task_guidance = self.task_guidance()
        self._history = []
        self._reward_fns = []
        self._pre_info_dict = None
        self._timesteps = 0
        self.conditioned_success = 0.0

        self._success_criteria = [
            lambda curr_info: self.is_task_conditions_met(curr_info)
        ]
        self.stop_criterions = [lambda _: self._timesteps >= 100]
        self.task_spec = None
        self.target_status = kwargs.get("target_status", None)
        self.task_type = kwargs.get("task_type", None)

    @classmethod
    def create(cls, task_cfg, task_data=None):
        if not task_data:
            path = "./data/alfred/alfred_test"
            tasks = load_data(path, "test")
            task_data = tasks[0]

        env = Alfred2DEnv.create(task_cfg["env"])
        world_confg = deepcopy(task_cfg["world"])
        if "world_data" in task_data.get("data"):
            world_confg.update({"data": task_data["data"]["world_data"]})

        env.create_world(world_confg)

        env.set_feedback_builder(TemplateBuilder(task_cfg["template"]))
        env.set_task_def(task_data["task_description"])
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
                        "task_description": task_data["task_description"],
                        "expert_actions": task_data["expert_actions"],
                    }
                )
                env.add_agent(agent)
        task = cls(
            env=env,
            template=task_cfg["template"],
            name=task_cfg.get("name", task_cfg["task"]),
            target_status=task_data["target_status"],
            task_type=task_data["task_type"],
        )
        task.task_spec = task_data.get("task_description")
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
        # self.env.reset()
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
            action_dict = {agent: action_dict for agent in self.env.agents.keys()}

        obs, reward, done, info = self.env.step(action_dict)
        self._timesteps += 1
        reward = self._compute_reward_hook(info)
        self._is_successful = self._determine_success(info)
        # done = self.env.is_terminated or self._is_successful
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

        while True:
            action_dict = dict()
            if info:
                logger.info(info)
                # agnt_info = info["state"]
                agnt_info = info["n"][0]
                agent_id = agnt_info["agent"]
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

                if agnt_info.get("feedback") is not None:
                    if render:
                        logger.emit(
                            {"role": "system", "content": agnt_info["feedback"]}
                        )
                    action_dict = {"prompt": agnt_info["feedback"]}

            if self._determine_stop(info):
                logger.emit(
                    {"role": "system", "content": str(self.conditioned_success)}
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
            self.env.update_object_props_after_action()

        if render:
            logger.emit({"role": "system", "content": "DONE!"})
            logger.emit("")
        return done

    def look_at_obj_in_light_conditions_met(self):
        ts = 2
        s = 0

        target_object_class = self.target_status["object_target"]
        agent_id = list(self.env.agents.keys())[0]
        agent = self.env.agents[agent_id]
        inventory_list = agent.inventory
        for inventory in inventory_list:
            object_class = inventory.props["objectId"].split("|")[0]
            if object_class == target_object_class:
                s += 1
                break
        observation = self.get_observation(agent)
        object_name_list = extract_object_names(observation)
        for object_name in object_name_list:
            if "lamp" in object_name:
                object_id = self.env.object_name2id[object_name]
                if self.env.world.get_object(object_id).props["isToggled"]:
                    s += 1
        return s, ts

    def pick_and_place_simple_conditions_met(self):
        ts = 1
        s = 0

        targets = self.target_status
        # print(targets)
        receptacles = self.get_objects_with_name_and_prop(
            targets["parent_target"], "receptacle"
        )
        pickupables = self.get_objects_with_name_and_prop(
            targets["object_target"], "pickupable"
        )
        # print(receptacles)
        # print(pickupables)
        # check if object needs to be sliced
        if targets["object_sliced"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p.id]) >= 1:
                s += 1

        try:
            found = 0
            for p in pickupables:
                for r in receptacles:
                    if r.get_child(p.id) is not None:
                        found += 1
            if found > 0:
                s += 1
            # if np.any(
            #     [
            #         np.any([p.id for r in receptacles if r.get_child(p.id) is not None])
            #         for p in pickupables
            #     ]
            # ):
            #     s += 1
        except:
            # TODO
            pass

        return s, ts

    def pick_two_obj_and_place_conditions_met(self):
        ts = 2
        s = 0

        targets = self.target_status
        receptacles = self.get_objects_with_name_and_prop(
            targets["parent_target"], "receptacle"
        )
        pickupables = self.get_objects_with_name_and_prop(
            targets["object_target"], "pickupable"
        )

        # check if object needs to be sliced
        if targets["object_sliced"]:
            ts += 2
            s += min(len([p for p in pickupables if "Sliced" in p.id]), 2)

        # placing each object counts as a goal_condition
        s += min(
            np.max(
                [
                    sum(
                        [1 if r.get_child(p.id) is not None else 0 for p in pickupables]
                    )
                    for r in receptacles
                ]
            ),
            2,
        )
        return s, ts

    def pick_and_place_with_movable_recep_conditions_met(self):
        ts = 3
        s = 0

        targets = self.target_status
        receptacles = self.get_objects_with_name_and_prop(
            targets["parent_target"], "receptacle"
        )
        pickupables = self.get_objects_with_name_and_prop(
            targets["object_target"], "pickupable"
        )
        movables = self.get_objects_with_name_and_prop(
            targets["mrecep_target"], "pickupable"
        )

        # check if object needs to be sliced
        if targets["object_sliced"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p.id]) >= 1:
                s += 1

        pickup_in_place = [
            p for p in pickupables for m in movables if m.get_child(p.id) is not None
        ]
        movable_in_place = [
            m for m in movables for r in receptacles if r.get_child(m.id) is not None
        ]
        # check if the object is in the final receptacle
        if len(pickup_in_place) > 0:
            s += 1
        # check if the movable receptacle is in the final receptacle
        if len(movable_in_place) > 0:
            s += 1
        # check if both the object and movable receptacle stack is in the final receptacle
        select_movables = [
            m for p in pickupables for m in movables if m.get_child(p.id) is not None
        ]
        select_receptacles = [
            r
            for m in select_movables
            for r in receptacles
            if r.get_child(m.id) is not None
        ]

        if len(select_receptacles) > 0:
            s += 1

        return s, ts

    def pick_heat_then_place_in_recep_conditions_met(self):
        ts = 3
        s = 0

        targets = self.target_status
        receptacles = self.get_objects_with_name_and_prop(
            targets["parent_target"], "receptacle"
        )
        pickupables = self.get_objects_with_name_and_prop(
            targets["object_target"], "pickupable"
        )

        # check if object needs to be sliced
        if targets["object_sliced"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p.id]) >= 1:
                s += 1

        objs_in_place = [
            p.id
            for p in pickupables
            for r in receptacles
            if r.get_child(p.id) is not None
        ]
        objs_heated = [p.id for p in pickupables if p.id in self.env.heated_objects]
        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was heated
        if len(objs_heated) > 0:
            s += 1
        # check if the object is both in the receptacle and hot
        if np.any([obj_id in objs_heated for obj_id in objs_in_place]):
            s += 1
        # print(ts, s)
        # print(targets)
        return s, ts

    def pick_cool_then_place_in_recep_conditions_met(self):
        ts = 3
        s = 0

        targets = self.target_status
        receptacles = self.get_objects_with_name_and_prop(
            targets["parent_target"], "receptacle"
        )
        pickupables = self.get_objects_with_name_and_prop(
            targets["object_target"], "pickupable"
        )

        # check if object needs to be sliced
        if targets["object_sliced"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p.id]) >= 1:
                s += 1

        objs_in_place = [
            p.id
            for p in pickupables
            for r in receptacles
            if r.get_child(p.id) is not None
        ]
        objs_cooled = [p.id for p in pickupables if p.id in self.env.cooled_objects]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was cooled
        if len(objs_cooled) > 0:
            s += 1
        # check if the object is both in the receptacle and hot
        if np.any([obj_id in objs_cooled for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def pick_clean_then_place_in_recep_conditions_met(self):
        ts = 3
        s = 0

        targets = self.target_status
        receptacles = self.get_objects_with_name_and_prop(
            targets["parent_target"], "receptacle"
        )
        pickupables = self.get_objects_with_name_and_prop(
            targets["object_target"], "pickupable"
        )

        # check if object needs to be sliced
        if targets["object_sliced"]:
            ts += 1
            if len([p for p in pickupables if "Sliced" in p.id]) >= 1:
                s += 1

        objs_in_place = [
            p.id
            for p in pickupables
            for r in receptacles
            if r.get_child(p.id) is not None
        ]
        objs_cleaned = [p.id for p in pickupables if p.id in self.env.cleaned_objects]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was cleaned
        if len(objs_cleaned) > 0:
            s += 1
        # check if the object is both in the receptacle and hot
        if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def get_objects_with_name_and_prop(self, name, prop):
        objs = []
        for obj_id, obj in self.env.world.objects.items():
            if name in obj_id and obj.props[prop]:
                objs.append(obj)
            children = obj.find_all_children()
            for child in children:
                if name in child.id and child.props[prop]:
                    objs.append(child)
        return objs

    def is_task_conditions_met(self, curr_info):
        def func_not_found():
            raise BaseException(
                "Task type error! Task Type :{} ".format(self.task_type)
            )

        conditions_met_funcs = {
            "pick_heat_then_place_in_recep": self.pick_heat_then_place_in_recep_conditions_met,
            "look_at_obj_in_light": self.look_at_obj_in_light_conditions_met,
            "pick_and_place_simple": self.pick_and_place_simple_conditions_met,
            "pick_cool_then_place_in_recep": self.pick_cool_then_place_in_recep_conditions_met,
            "pick_two_obj_and_place": self.pick_two_obj_and_place_conditions_met,
            "pick_and_place_with_movable_recep": self.pick_and_place_with_movable_recep_conditions_met,
            "pick_clean_then_place_in_recep": self.pick_clean_then_place_in_recep_conditions_met,
        }
        # print(self.task_type)
        s, ts = conditions_met_funcs.get(self.task_type, func_not_found)()
        self.conditioned_success = round(s / ts, 4)
        return s == ts


def extract_object_names(sentence):
    pattern = r"\b(\w+_\d+)\b"
    object_ids = re.findall(pattern, sentence)
    return object_ids
