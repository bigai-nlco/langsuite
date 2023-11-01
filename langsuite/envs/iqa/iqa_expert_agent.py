# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import random
import re
from copy import deepcopy

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.agents.simple_agent import SimpleAgent
from langsuite.shapes import Vector2D
from langsuite.utils.grid_world import GridWorld, cal_wall_min_max, get_direction
from langsuite.utils.logging import logger


@AGENT_REGISTRY.register()
class IqaExpertAgent(SimpleAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config=agent_config)
        self.status = dict(started=False)
        self.chat_history = []
        self.question_type = None
        self.object_class = None
        self.current_object_id = None
        self.current_recep_id = None
        self.recep = None
        self.unseen_object_list_all = []
        self.open_list_all = []
        self.unseen_object_list = []
        self.open_list = []
        self.only_category_flag = False
        self.target_status = agent_config.get("target_status")
        self.start_status = agent_config.get("start_status")
        self.already_seen_object_list = set()
        self.first_go = True
        self.last_action = ""
        self.plans = []
        self.max_view_distance = None
        self.max_manipulate_distance = None
        self.view_degree = None
        self.actions = ["YES."]

    def set_env(self, env):
        self.env = env
        self.question_type = self.env.question_type
        self.object_class = self.env.question_info["object_class"]
        if "recept" in self.env.question_info:
            self.recep = self.env.question_info["recept"]
        self.generate_plan()

    def generate_plan(self):
        nodes = []
        if self.question_type == 0:
            if not self.env.answer:
                first_flag = True
                for object_id, obj in self.env.world.objects.items():
                    sample_num = random.randint(0, 15)
                    if not sample_num:
                        if (
                            not obj.props["parentReceptacles"]
                            or (
                                obj.props["parentReceptacles"]
                                and "Floor" in obj.props["parentReceptacles"][0]
                            )
                            or (
                                obj.props["parentReceptacles"]
                                and not self.env.world.objects[
                                    obj.props["parentReceptacles"][0]
                                ].props["openable"]
                            )
                        ):
                            if first_flag:
                                if obj.props["openable"]:
                                    node = {
                                        "name": object_id,
                                        "object_id": object_id,
                                        "position": {
                                            "x": obj.position.x,
                                            "y": obj.position.y,
                                        },
                                        "action": "Open({})".format(
                                            self.env.object_id2name[object_id]
                                        ),
                                    }
                                    nodes.append(node)

                                else:
                                    first_flag = False
                                    node = {
                                        "name": object_id,
                                        "object_id": object_id,
                                        "position": {
                                            "x": obj.position.x,
                                            "y": obj.position.y,
                                        },
                                        "action": "",
                                    }
                                    nodes.append(node)
                            else:
                                self.unseen_object_list_all.append(
                                    [self.object_class + "_0"]
                                )
                                self.open_list_all.append([])
                                node = {
                                    "name": object_id,
                                    "object_id": object_id,
                                    "position": {
                                        "x": obj.position.x,
                                        "y": obj.position.y,
                                    },
                                    "action": "",
                                }
                                nodes.append(node)
            if self.env.answer:
                for object_id, obj in self.env.world.objects.items():
                    object_category = object_id.split("|")[0]
                    if object_category == self.object_class:
                        if (
                            not obj.props["parentReceptacles"]
                            or (
                                obj.props["parentReceptacles"]
                                and "Floor" in obj.props["parentReceptacles"][0]
                            )
                            or (
                                obj.props["parentReceptacles"]
                                and obj.props["parentReceptacles"][0]
                                in self.env.world.objects
                                and not self.env.world.objects[
                                    obj.props["parentReceptacles"][0]
                                ].props["openable"]
                            )
                        ):
                            self.only_category_flag = True
                            self.unseen_object_list_all.append([object_id])
                            self.open_list_all.append([])
                            self.unseen_object_list_all.append([object_id])
                            self.open_list_all.append([])
                            self.unseen_object_list = self.unseen_object_list_all.pop(0)
                            self.open_list = self.open_list_all.pop(0)
                            node = {
                                "name": object_id,
                                "object_id": object_id,
                                "position": {"x": obj.position.x, "y": obj.position.y},
                                "action": "",
                            }
                            nodes.append(node)
                            self.plans = nodes
                        else:
                            if not self.env.world.objects[
                                obj.props["parentReceptacles"][0]
                            ].props["openable"]:
                                self.unseen_object_list.append(object_id)
                                self.current_object_id = object_id
                                node = {
                                    "name": object_id,
                                    "object_id": object_id,
                                    "position": {
                                        "x": obj.position.x,
                                        "y": obj.position.y,
                                    },
                                    "action": "",
                                }
                                nodes.append(node)
                            else:
                                t = random.choice(list(self.env.world.objects.keys()))
                                recep_obj = self.env.world.objects[
                                    obj.props["parentReceptacles"][0]
                                ]
                                node = {
                                    "name": obj.props["parentReceptacles"][0],
                                    "object_id": obj.props["parentReceptacles"][0],
                                    "position": {
                                        "x": recep_obj.position.x,
                                        "y": recep_obj.position.y,
                                    },
                                    "action": "Open({})".format(
                                        self.env.object_id2name[
                                            obj.props["parentReceptacles"][0]
                                        ]
                                    ),
                                }
                                nodes.append(node)
                                node = {
                                    "name": t,
                                    "object_id": t,
                                    "position": {
                                        "x": self.env.world.objects[t].position.x,
                                        "y": self.env.world.objects[t].position.y,
                                    },
                                    "action": "",
                                }
                                nodes.insert(0, node)
        elif self.question_type == 1:
            if not self.env.answer:
                first_flag = True
                for object_id, obj in self.env.world.objects.items():
                    sample_num = random.randint(0, 15)
                    if not sample_num:
                        if (
                            not obj.props["parentReceptacles"]
                            or (
                                obj.props["parentReceptacles"]
                                and "Floor" in obj.props["parentReceptacles"][0]
                            )
                            or (
                                obj.props["parentReceptacles"]
                                and not self.env.world.objects[
                                    obj.props["parentReceptacles"][0]
                                ].props["openable"]
                            )
                        ):
                            if first_flag:
                                if obj.props["openable"]:
                                    first_flag = False
                                    self.unseen_object_list_all.append(
                                        [object_id, self.object_class + "_0"]
                                    )
                                    self.open_list_all.append([object_id])
                                    self.unseen_object_list_all.append(
                                        [object_id, self.object_class + "_0"]
                                    )
                                    self.open_list_all.append([object_id])
                                    self.unseen_object_list = (
                                        self.unseen_object_list_all.pop(0)
                                    )
                                    self.open_list = self.open_list_all.pop(0)
                                    node = {
                                        "name": object_id,
                                        "object_id": object_id,
                                        "position": {
                                            "x": obj.position.x,
                                            "y": obj.position.y,
                                        },
                                        "action": "Open({})".format(
                                            self.env.object_id2name[object_id]
                                        ),
                                    }
                                    nodes.append(node)

                                else:
                                    first_flag = False
                                    self.unseen_object_list_all.append(
                                        [self.object_class + "_0"]
                                    )
                                    self.open_list_all.append([])
                                    self.unseen_object_list_all.append(
                                        [self.object_class + "_0"]
                                    )
                                    self.open_list_all.append([])
                                    self.unseen_object_list = (
                                        self.unseen_object_list_all.pop(0)
                                    )
                                    self.open_list = self.open_list_all.pop(0)
                                    node = {
                                        "name": object_id,
                                        "object_id": object_id,
                                        "position": {
                                            "x": obj.position.x,
                                            "y": obj.position.y,
                                        },
                                        "action": "",
                                    }
                                    nodes.append(node)
                            else:
                                self.unseen_object_list_all.append(
                                    [self.object_class + "_0"]
                                )
                                self.open_list_all.append([])
                                node = {
                                    "name": object_id,
                                    "object_id": object_id,
                                    "position": {
                                        "x": obj.position.x,
                                        "y": obj.position.y,
                                    },
                                    "action": "",
                                }
                                nodes.append(node)
            if self.env.answer:
                for object_id, obj in self.env.world.objects.items():
                    object_category = object_id.split("|")[0]
                    if object_category == self.recep:
                        self.only_category_flag = True
                        node = {
                            "name": object_id,
                            "object_id": object_id,
                            "position": {"x": obj.position.x, "y": obj.position.y},
                            "action": "Open({})".format(
                                self.env.object_id2name[object_id]
                            ),
                        }
                        nodes.append(node)
        elif self.question_type == 2:
            if not self.env.answer:
                pass
            if self.env.answer:
                for object_id, obj in self.env.world.objects.items():
                    object_category = object_id.split("|")[0]
                    if object_category == self.object_class:
                        if (
                            not obj.props["parentReceptacles"]
                            or (
                                obj.props["parentReceptacles"]
                                and "Floor" in obj.props["parentReceptacles"][0]
                            )
                            or (
                                obj.props["parentReceptacles"]
                                and not self.env.world.objects[
                                    obj.props["parentReceptacles"][0]
                                ].props["openable"]
                            )
                        ):
                            node = {
                                "name": object_id,
                                "object_id": object_id,
                                "position": {"x": obj.position.x, "y": obj.position.y},
                                "action": "",
                            }
                            nodes.append(node)
                        else:
                            recep_obj = self.env.world.objects[
                                obj.props["parentReceptacles"][0]
                            ]
                            node = {
                                "name": object_id,
                                "object_id": object_id,
                                "position": {
                                    "x": recep_obj.position.x,
                                    "y": recep_obj.position.y,
                                },
                                "action": "Open({})".format(
                                    self.env.object_id2name[
                                        obj.props["parentReceptacles"][0]
                                    ]
                                ),
                            }
                            nodes.append(node)
                # random seach
                for object_id, obj in self.env.world.objects.items():
                    if object_category != self.object_class:
                        sample_num = random.randint(0, 15)
                        if not sample_num:
                            if (
                                not obj.props["parentReceptacles"]
                                or (
                                    obj.props["parentReceptacles"]
                                    and "Floor" in obj.props["parentReceptacles"][0]
                                )
                                or (
                                    obj.props["parentReceptacles"]
                                    and not self.env.world.objects[
                                        obj.props["parentReceptacles"][0]
                                    ].props["openable"]
                                )
                            ):
                                node = {
                                    "name": object_id,
                                    "object_id": object_id,
                                    "position": {
                                        "x": obj.position.x,
                                        "y": obj.position.y,
                                    },
                                    "action": "",
                                }
                                nodes.append(node)
                            else:
                                recep_obj = self.env.world.objects[
                                    obj.props["parentReceptacles"][0]
                                ]
                                node = {
                                    "name": object_id,
                                    "object_id": object_id,
                                    "position": {
                                        "x": recep_obj.position.x,
                                        "y": recep_obj.position.y,
                                    },
                                    "action": "Open({})".format(
                                        self.env.object_id2name[
                                            obj.props["parentReceptacles"][0]
                                        ]
                                    ),
                                }
                                nodes.append(node)

        self.plans = nodes

    def query_expert(self):
        action = ""
        if self.question_type == 0:
            if len(self.actions) > 0 and self.actions[0] != "Start":
                observation = self.env.get_observation(self)
                if self.object_class.lower() in observation:
                    return "answer [True]"
                action = self.actions.pop(0)
                self.last_action = action
            elif len(self.actions) == 0 and not self.plans:
                return "answer [False]"
            else:
                if self.plans:
                    plan = self.plans.pop(0)
                    object_id = plan["object_id"]
                    room_polygons = self.env.world.room_polygons
                    x_min, x_max, y_min, y_max = cal_wall_min_max(room_polygons)
                    grid_world = GridWorld(x_min, x_max, y_min, y_max, self.step_size)
                    agent_direction = get_direction(self.view_vector)
                    for obj_id, obj in self.env.world.objects.items():
                        if "Floor" not in obj_id:
                            x_list = list(obj.geometry.shapely_geo.exterior.xy[0][:4])
                            y_list = list(obj.geometry.shapely_geo.exterior.xy[1][:4])

                            grid_world.add_object(
                                obj.position.x, obj.position.y, (x_list, y_list)
                            )
                    object = self.env.world.objects.get(object_id)
                    target_position = (object.position.x, object.position.y)
                    _, action_trajectory, _ = grid_world.get_path(
                        self.position,
                        target_position,
                        agent_direction,
                        grid_world.grid,
                    )
                    if not action_trajectory:
                        action_trajectory = []
                    action_trajectory.append("TurnRight")
                    action_trajectory.append("TurnRight")
                    action_trajectory.append("TurnRight")
                    if plan["action"]:
                        object_id = parse_method(plan["action"], "Open")
                        action_trajectory.append(plan["action"])
                    action_trajectory.append("TurnRight")
                    self.actions.extend(action_trajectory)
                    action = self.actions.pop(0)
                    self.last_action = action
        elif self.question_type == 1:
            if len(self.actions) > 0 and self.actions[0] != "Start":
                observation = self.env.get_observation(self)
                if "Open" in self.last_action:
                    observation = self.env.get_openned_object_observation(
                        self.env.object_name2id[parse_method(self.last_action, "Open")]
                    )
                    if (
                        self.object_class.lower() in observation
                        and "In/on it" in observation
                    ):
                        return "answer [True]"
                action = self.actions.pop(0)
                self.last_action = action
            elif len(self.actions) == 0 and not self.plans:
                return "answer [False]"
            else:
                if self.plans:
                    plan = self.plans.pop(0)
                    object_id = plan["object_id"]
                    room_polygons = self.env.world.room_polygons
                    x_min, x_max, y_min, y_max = cal_wall_min_max(room_polygons)
                    grid_world = GridWorld(x_min, x_max, y_min, y_max, self.step_size)
                    agent_direction = get_direction(self.view_vector)
                    for obj_id, obj in self.env.world.objects.items():
                        if "Floor" not in obj_id:
                            x_list = list(obj.geometry.shapely_geo.exterior.xy[0][:4])
                            y_list = list(obj.geometry.shapely_geo.exterior.xy[1][:4])

                            grid_world.add_object(
                                obj.position.x, obj.position.y, (x_list, y_list)
                            )
                    object = self.env.world.objects.get(object_id)
                    target_position = (object.position.x, object.position.y)
                    _, action_trajectory, _ = grid_world.get_path(
                        self.position,
                        target_position,
                        agent_direction,
                        grid_world.grid,
                    )
                    if not action_trajectory:
                        action_trajectory = []
                    action_trajectory.append("TurnRight")
                    action_trajectory.append("TurnRight")
                    action_trajectory.append("TurnRight")
                    if plan["action"]:
                        object_id = parse_method(plan["action"], "Open")
                        action_trajectory.append(plan["action"])
                    action_trajectory.append("TurnRight")
                    self.actions.extend(action_trajectory)
                    action = self.actions.pop(0)
                    self.last_action = action
        elif self.question_type == 2:
            if len(self.actions) > 0 and self.actions[0] != "Start":
                observation = self.env.get_observation(self)
                if "Open" == self.last_action:
                    observation = self.env.get_openned_object_observation(
                        self.env.object_name2id[parse_method(self.last_action, "Open")]
                    )
                    if self.object_class.lower() in observation:
                        object_list = extract_object_names(observation)
                        for obj_name in object_list:
                            if (
                                self.object_class.lower() in obj_name
                                and obj_name not in self.already_seen_object_list
                            ):
                                self.env.count_number += 1
                                self.already_seen_object_list.add(obj_name)
                                if self.env.count_number == self.env.answer:
                                    return "answer [{}]".format(self.env.count_number)
                else:
                    if self.object_class.lower() in observation:
                        object_list = extract_object_names(observation)
                        for obj_name in object_list:
                            if (
                                self.object_class.lower() in obj_name
                                and obj_name not in self.already_seen_object_list
                            ):
                                self.env.count_number += 1
                                self.already_seen_object_list.add(obj_name)
                                if self.env.count_number == self.env.answer:
                                    return "answer [{}]".format(self.env.count_number)
                action = self.actions.pop(0)
                self.last_action = action
            elif len(self.actions) == 0 and not self.plans:
                return "answer [{}]".format(self.env.count_number)
            else:
                if self.plans:
                    plan = self.plans.pop(0)
                    object_id = plan["object_id"]
                    room_polygons = self.env.world.room_polygons
                    x_min, x_max, y_min, y_max = cal_wall_min_max(room_polygons)
                    grid_world = GridWorld(x_min, x_max, y_min, y_max, self.step_size)
                    agent_direction = get_direction(self.view_vector)
                    for obj_id, obj in self.env.world.objects.items():
                        if "Floor" not in obj_id:
                            x_list = list(obj.geometry.shapely_geo.exterior.xy[0][:4])
                            y_list = list(obj.geometry.shapely_geo.exterior.xy[1][:4])

                            grid_world.add_object(
                                obj.position.x, obj.position.y, (x_list, y_list)
                            )
                    if "Drop" in plan["action"]:
                        target_position = (plan["position"]["x"], plan["position"]["y"])
                    else:
                        object = self.env.world.objects.get(object_id)
                        target_position = (object.position.x, object.position.y)
                    _, action_trajectory, _ = grid_world.get_path(
                        self.position,
                        target_position,
                        agent_direction,
                        grid_world.grid,
                    )
                    if not action_trajectory:
                        action_trajectory = []
                    action_trajectory.append("TurnRight")
                    action_trajectory.append("TurnRight")
                    action_trajectory.append("TurnRight")
                    if plan["action"]:
                        object_id = parse_method(plan["action"], "Open")
                        action_trajectory.append(plan["action"])
                    action_trajectory.append("TurnRight")
                    self.actions.extend(action_trajectory)
                    action = self.actions.pop(0)
                    self.last_action = action

        return action

    def get_observation_diff(self, starts, targets):
        diff = []
        for start_obj, target_obj in zip(starts, targets):
            assert start_obj["name"] == target_obj["name"]
            if start_obj["position"] != target_obj["position"]:
                action_start = deepcopy(start_obj)
                action_target = deepcopy(target_obj)
                action_start["action"] = "Pickup"
                action_target["action"] = "Drop"
                diff.append(action_start)
                action_target["object_id"] = action_start["object_id"]
                diff.append(action_target)
            if "openness" in target_obj:
                action_open = deepcopy(target_obj)
                action_open["object_id"] = start_obj["object_id"]
                action_open["action"] = "Open"
                diff.append(action_open)
        return diff

    def step(self, action_dict):
        parsed_response = {}
        expert_action = self.query_expert()
        if not self.plans and expert_action == "Stop":
            expert_action = "answer [True]"
        parsed_response = self.parse_expert_action(expert_action)
        logger.info(parsed_response)
        success = True
        if "action" in parsed_response and parsed_response["action"] != "UserInput":
            if parsed_response["action"] == "Pass":
                parsed_response["feedback"] = self.env.feedback_builder.build("Pass")
                success = False
            elif parsed_response["action"] == "Stop":
                parsed_response["feedback"] = self.env.feedback_builder.build("Stop")
                success = True
            elif parsed_response["action"] == "Open":
                parsed_response["action_arg"]["object_id"] = self.env.object_name2id[
                    parsed_response["action_arg"]["object_name"]
                ]
                action_status = self.execute(
                    action=parsed_response["action"],
                    **parsed_response["action_arg"],
                )
                logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
            else:
                action_status = self.execute(
                    action=parsed_response["action"],
                    **parsed_response["action_arg"],
                )
                logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
        return success, parsed_response

    def execute(self, *, action: str = None, **action_kwargs):
        logger.info(f"Working on action {action}")
        if not self.is_valid_action(action):
            logger.info(f"Invalid action: {action}")
            return ActionFeedback(success=False, feedback=f"Invalid action: {action}")

        action_or = get_action(action_name=action, env=self.env, agent=self)
        if action_or:
            return action_or.step(**action_kwargs)
        else:
            logger.info(f"Action {action} not found in environment.")
            return ActionFeedback(
                success=False, feedback=f"Action {action} not found in environment."
            )

    def parse_expert_action(self, response):
        """
        if there exists an action in the response, then excecute response and return action feedback
        """
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response, feedback=self.env.feedback_builder.build("intro")
                )
            else:
                self.status["started"] = True
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "IqaStart",
                        task=self.env.get_task_def(),
                        object_str=self.env.get_observation(self),
                    ),
                )
        elif "Answer" in response:
            response_answer = parse_answer(response)
            gold_answer = self.env.get_answer()
            return dict(
                response=response,
                action="Answer",
                answer_info="You are Right!"
                if response_answer == gold_answer
                else "You are Wrong!",
                action_arg={},
            )
        else:
            if "MoveAhead" in response:
                action = "MoveAhead"
                response = "move_ahead"
            elif "MoveBack" in response:
                action = "MoveBack"
            elif "TurnLeft" in response:
                action = "TurnLeft"
                response = "turn_left"
            elif "TurnRight" in response:
                action = "TurnRight"
                response = "turn_right"
            elif "Open" in response:
                action = "Open"
                object_name = parse_method(response, "Open")
                response = "open[{}]".format(object_name)
                return dict(
                    response=response,
                    action=action,
                    action_arg={"object_name": object_name},
                )
            elif "Stop" in response:
                action = "Stop"
            else:
                action = "Pass"
            return dict(response=response, action=action, action_arg={})

    def step_message(self, message=None):
        actions = self.message_parser.parse(message["content"])
        if len(actions) > 0:
            logger.debug(f"Parsed results: {actions}")
            if len(actions) == 1:
                action = actions[0]
                return self.step(
                    action=action["action"],
                    action_args=action["args"],
                    **action["kwargs"],
                )
            results = []
            for action in actions:
                results.append(
                    self.step(
                        action=action["action"],
                        action_args=action["args"],
                        **action["kwargs"],
                    )
                )
            return results
        return ActionFeedback(
            success=False, feedback="No valid action found in message."
        )

    def reset(self):
        self.view_vector = Vector2D(0, 1)
        self.inventory.clear()
        self.set_config(self.init_cfg)

    def set_name(self, name):
        self.name = name


def parse_answer(response):
    pattern = r"Robot\.Answer\((.*?)\)"
    matches = re.findall(pattern, response)

    for match in matches:
        return match
    return ""


def parse_method(response, method):
    pattern = r"" + method + "\((.*?)\)"
    matches = re.findall(pattern, response)

    for match in matches:
        return match
    return ""


def extract_object_names(sentence):
    pattern = r"\b(\w+_\d+)\b"
    object_ids = re.findall(pattern, sentence)
    return object_ids
