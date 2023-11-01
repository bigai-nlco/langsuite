# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.agents.simple_agent import SimpleAgent
from langsuite.llms import create_llm, create_llm_prompts, process_llm_results
from langsuite.llms.output_parsers import RegexOutputParser
from langsuite.shapes import Cone2D, Point2D, Polygon2D, Vector2D
from langsuite.utils import math_utils
from langsuite.utils.logging import logger
from langsuite.utils.string_utils import camelcase


@AGENT_REGISTRY.register()
class TeachAgent(SimpleAgent):
    """
    TEACh agent class

    This class provides functions to:
        - Generate agent action, get agent observation.
    """

    def __init__(self, agent_config: Dict) -> None:
        super().__init__(agent_config=agent_config)
        self.isExpert = agent_config.get("isExpert", False)
        self.agent_id = agent_config["agent_id"]
        self.opponent_agent_id = 1 - self.agent_id
        self.agent_name = "commander" if self.agent_id == 0 else "follower"
        self.oppo_name = "commander" if self.agent_id == 1 else "follower"

        self.position = Point2D(agent_config.get("position"))
        self.set_config(agent_config)
        self.view_vector = Vector2D(0, 1)
        self.view_geometry = self._compute_fov_geometry()
        self.inventory = []

        self.status = dict(started=False)
        self.chat_history = []
        self.task_description = agent_config.get("task_description")
        self.llm = create_llm(agent_config.get("llm"))
        self.output_parser = RegexOutputParser(RegexOutputParser.ALFRED_ACTION_REGEX)
        self.current_prompt = None
        logger.info(f"Successfully add agent: {self.cfg}")

    @classmethod
    def create(cls, agent_cfg: Dict):
        return cls(agent_config=agent_cfg)

    def step(self, action_dict):
        if type(action_dict) == str:
            action_dict = dict(prompt=action_dict)

        prompt = action_dict.get("prompt")
        self.current_prompt = prompt
        parsed_response = {}
        if self.isExpert:
            success = True
            response = action_dict["action"]
            parsed_response = self.parse_expert_action(action_dict)
        else:
            response = self.fetch_prompt_response(prompt)
            parsed_response = self.parse_response(response)
            logger.info(parsed_response)
            success = parsed_response.get("success", True)

        if (
            success
            and "action" in parsed_response
            and parsed_response["action"] != "UserInput"
        ):
            if parsed_response["action"] == "Stop":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "Stop", "success"
                )
                success = True
                action_status = self.execute(
                    action=parsed_response["action"],
                    **parsed_response["action_arg"],
                )
                logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
        if parsed_response.get("action", None) in ["Chat"]:
            oppo_agent = self.env.agents[self.opponent_agent_id]
            oppo_agent.chat_history += [
                {
                    "role": "assistant",
                    "content": f"{self.agent_name}: "
                    + parsed_response["action_arg"]["chat_response"],
                },
            ]
        elif parsed_response.get("action", None) in ["SelectOid"]:
            oppo_agent = self.env.agents[self.opponent_agent_id]
            oppo_agent.chat_history += [
                {
                    "role": "assistant",
                    "content": response,
                },
            ]

        def convert_action_name(action_name):
            words = re.findall("[A-Z][^A-Z]*", action_name)
            return "_".join(words).lower()

        if (
            self.status["started"] is True
            and "action" in parsed_response
            and "action_arg" in parsed_response
            and "chat_response" in parsed_response["action_arg"]
        ):
            formated_response = (
                convert_action_name(parsed_response["action"])
                + " "
                + "["
                + parsed_response["action_arg"]["chat_response"]
                + "]"
            )
        else:
            formated_response = response

        self.chat_history += [
            {"role": "system", "content": prompt},
            {"role": "assistant", "content": formated_response},
        ]
        self.status["started"] = True
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
                success=False,
                feedback=f"Action {action} not found in environment.",
            )

    def _get_obj_id_by_name(self, name, oracle=False):
        """Get object id by name"""
        name = name.lower()
        if oracle:
            objects_in_view = self.env.world.id2object
        else:
            objects_in_view = self.env.get_observed_objects(agent=self)
        # possible_objects = []
        objname_patern = re.compile(r"[a-zA-Z]+_[0-9]")
        match = objname_patern.match(name)
        if match and name in self.env.object_name2id:
            target_id = self.env.object_name2id[name]
            if target_id in objects_in_view:
                return target_id
        else:
            possible_objects = []
            for id, o in objects_in_view.items():
                iter_type = camelcase(id.split("|")[0]).lower()
                if iter_type in name:
                    pose_info = o.get_obj_pose_info()
                    pose_info["distance"] = math_utils.euclidean_distance(
                        self.position, o.position
                    )
                    possible_objects.append(pose_info)
                for simbotObjectClass in o.props["simbotObjectClass"]:
                    if simbotObjectClass.lower() in name:
                        pose_info = o.get_obj_pose_info()
                        pose_info["distance"] = math_utils.euclidean_distance(
                            self.position, o.position
                        )
                        possible_objects.append(pose_info)
                        break
            if len(possible_objects) > 0:
                possible_objects = sorted(
                    possible_objects, key=lambda po: (po["distance"], po["name"])
                )
                return possible_objects[0]["objectId"]
        return None

    def _justify_action_name(self, name):
        name = name.lower()
        splits = name.split("_")
        return "".join([s[0].upper() + s[1:] for s in splits])

    def parse_expert_action(self, expert_action):
        if not self.status["started"]:
            return dict(
                response="Yes.",
                feedback=self.env.feedback_builder.build(
                    "Start",
                    task_description=self.task_description,
                    observation=self.env.get_observation(self),
                ),
            )
        self.env.expert_steps += 1
        action = expert_action["action"]
        action_kwargs = {}

        if "Forward" == action:
            action = "MoveAhead"
        elif "Backward" == action:
            action = "MoveBack"
        elif "Turn Left" == action:
            action = "TurnLeft"
        elif "Turn Right" == action:
            action = "TurnRight"
        elif "OpenProgressCheck" == action:
            pass
        elif "Text" == action:
            action = "Chat"
            action_kwargs["chat_response"] = expert_action["action_arg"][
                "corrected_utterance"
            ]
        elif "Pan Left" == action:
            action = "PanLeft"
        elif "Pan Right" == action:
            action = "PanRight"
        elif "SelectOid" == action:
            action_kwargs["object_id"] = expert_action["action_arg"]["query"]
        elif "SearchObject" == action:
            query = expert_action["action_arg"]["query"]
            action_kwargs["object_id"] = self._get_obj_id_by_name(
                query, self.agent_name == "commander"
            )
        elif "Pickup" == action:
            action = "PickUp"
            action_kwargs["object_id"] = expert_action["action_arg"]["oid"]
        elif action in [
            "Place",
            "Pour",
            "Slice",
            "ToggleOn",
            "ToggleOff",
            "Open",
            "Close",
        ]:
            action_kwargs["object_id"] = expert_action["action_arg"]["oid"]

        response = []
        convert_action_name = lambda action_name: "_".join(
            re.findall("[A-Z][^A-Z]*", action_name)
        ).lower()
        if action == "Chat":
            response.append(
                (
                    "act",
                    convert_action_name(action)
                    + " ["
                    + action_kwargs["chat_response"]
                    + "]",
                )
            )
        elif action == "SearchObject":
            object_id = action_kwargs.get("object_id", "")
            if object_id is None:
                object_id = ""
            converted_action_name = convert_action_name(action)
            if len(object_id) > 0:
                object_name = self.env.object_id2name[object_id]
                response.append(
                    ("act", converted_action_name + " [" + object_name + "]")
                )
            else:
                response.append(("act", converted_action_name))
            action = "SelectOid"
        else:
            object_id = action_kwargs.get("object_id", "")
            if object_id is None:
                object_id = ""
            converted_action_name = convert_action_name(action)
            if len(object_id) > 0:
                object_name = self.env.object_id2name[object_id]
                response.append(
                    ("act", converted_action_name + " [" + object_name + "]")
                )
            else:
                response.append(("act", converted_action_name))

        print(self.position, self.view_vector, action_kwargs.get("object_id", ""))
        return dict(
            response=response,
            action=action,
            action_arg=action_kwargs,
        )

    def parse_response(self, response):
        """
        if there exists an action in the response, then excecute response and return action feedback
        """
        obj_id = None
        response = [["act", response]]
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.max_view_distance / self.step_size,
                        degree=self.aov / 2,
                        max_inventory=self.inventory_capacity,
                        max_manipulation_steps=self.max_manipulate_distance
                        / self.step_size,
                    ),
                )
            else:
                self.status["started"] = True
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "Start",
                        task_description=self.task_description,
                        observation=self.env.get_observation(self),
                    ),
                )
        else:
            if "]" in response[0][1]:
                response[0][1] = response[0][1].split("]")[0] + "]"
            if ":" in response[0][1]:
                response[0][1] = response[0][1].split(":")[1]
            parsed_actions = self.output_parser.parse(response[0][1].lower())
            action_dicts = []
            action_without_arg = [
                "move_ahead",
                "move_back",
                "turn_left",
                "turn_right",
                "pan_left",
                "pan_right",
                "open_progress_check",
                "no_op",
                "stop",
            ]
            action_with_arg = [
                "pick_up",
                "place",
                "pour",
                "open",
                "close",
                "toggle_on",
                "toggle_off",
                "slice",
                "search_object",
                "select_oid",
                "go_to",
                "chat",
            ]

            for action_name, action_arg in parsed_actions:
                if action_name in action_without_arg:
                    action_dicts.append(
                        dict(action_name=self._justify_action_name(action_name))
                    )
                elif action_name in action_with_arg:
                    obj_name = action_arg.strip("'").strip('"')

                    if action_name == "search_object":
                        action_name = "select_oid"
                    elif action_name == "chat":
                        action_dicts.append(
                            dict(
                                action_name=self._justify_action_name(action_name),
                                action_arg={"chat_response": obj_name},
                            )
                        )
                        continue
                    oracle = self.agent_name == "commander"
                    obj_id = self._get_obj_id_by_name(obj_name, oracle)
                    if obj_id is None:
                        return dict(
                            success=False,
                            response=response,
                            feedback=self.env.feedback_builder.build(
                                "InvalidAction",
                                "failure.objectNotProvide",
                                object=obj_name,
                            ),
                        )
                    logger.info("obj name: {} obj id {}.".format(obj_name, obj_id))
                    action_dicts.append(
                        dict(
                            action_name=self._justify_action_name(action_name),
                            action_arg={
                                "object_id": obj_id,
                                "object_name": obj_name,
                                "chat_response": obj_name,
                            },
                        )
                    )
            if len(action_dicts) > 1:
                return dict(
                    success=False,
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "InvalidAction", "failure.multipleActions"
                    ),
                )
            elif len(action_dicts) == 0:
                return dict(
                    success=False,
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "InvalidAction", "failure.actionNotFound"
                    ),
                )

            return dict(
                response=response,
                action=action_dicts[0]["action_name"],
                action_arg=action_dicts[0].get("action_arg", {}),
            )

    def fetch_prompt_response(self, prompt):
        prompts = deepcopy(self.chat_history)
        prompts.append({"role": "system", "content": str(prompt)})

        response = self.llm(messages=create_llm_prompts(messages=prompts))
        logger.info(response)
        return process_llm_results(response)

    def get_inventory(self):
        if len(self.inventory) == 0:
            return "Empty"

        return ", ".join([self.env.object_id2name[o.id] for o in self.inventory])

    def reset(self):
        self.view_vector = Vector2D(0, 1)
        self.inventory.clear()
        self.set_config(self.init_cfg)

    def set_config(self, agent_config):
        self.cfg.update(agent_config)
        self.step_size = agent_config.get("step_size", 0.25)
        # self.valid_rotations = agent_config.get("rotation")
        self.focal_length = agent_config.get("focal_length", 10)
        self.aov = math_utils.compute_horizonal_aov(self.focal_length)
        self.max_view_distance = agent_config.get("max_view_distance", 10)
        self.max_manipulate_distance = agent_config.get("max_manipulate_distance", 1)
        self.inventory_capacity = agent_config.get("inventory_capacity", 1)
        if "rotation" in agent_config:
            self.view_vector.rotate(agent_config["rotation"])
        self.update()

    def update(self):
        self.view_geometry = self._compute_fov_geometry()

    def _compute_fov_geometry(self):
        return Cone2D(
            center=self.position,
            radius=self.max_view_distance,
            direction=self.view_vector,
            angle=self.aov,
        )

    # TODO  view obstables is not taken into consideration
    def can_manipulate(self, position: Point2D, manipulation_distance=None):
        if not manipulation_distance:
            manipulation_distance = self.max_manipulate_distance
        return (
            math_utils.euclidean_distance(position, self.position)
            <= manipulation_distance
        )

    # TODO  view obstables is not taken into consideration
    # TODO  other situations return False?
    def can_observe(self, geometry):
        if isinstance(geometry, Polygon2D):
            if self.view_geometry.intersects(geometry):
                return True
            position = geometry.centroid
        if isinstance(geometry, Point2D):
            position = geometry
            return math_utils.euclidean_distance(
                position, self.position
            ) <= self.max_view_distance and math_utils.angle_between_vectors(
                position - self.position, self.view_vector
            ) <= (
                self.aov / 2.0
            )
        return False

    def set_position(self, position: Point2D):
        self.position = position
        self.update()

    def rotate(self, angle):
        self.view_vector.rotate(angle)
        self.view_geometry.rotate(angle=angle)
        self.update()

    def get_rotation(self):
        a = (0, 1)
        b = (self.view_vector.x, self.view_vector.y)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        angle = math.acos(dot_product / (norm_a * norm_b))
        rotation = angle * 180 / math.pi
        if self.view_vector.x < 0:
            return 360 - rotation
        return rotation

    def add_inventory(self, inventory) -> bool:
        """
        Returns:
            Success or not.
        """
        if len(self.inventory) < self.inventory_capacity:
            self.inventory.append(inventory)
            return True
        return False

    def get_object_in_inventory(self, object_id: str):
        for obj in self.inventory:
            if obj.id == object_id:
                return obj
        return None

    def plot(self, axes=None):
        x, y = self.view_geometry.xy
        axes.fill(x, y, "w", alpha=0.8)
        axes.add_artist(
            plt.Circle(
                (self.position.x, self.position.y), 0.2, fill=True, color="orange"
            )
        )

    def render(self, fig=None):
        if not fig:
            fig = go.Figure()

        radius = 0.05
        svg = re.findall(r'd="(.*?)"', self.view_geometry.shapely_geo.svg())[0]
        fig.add_shape(
            type="path", path=svg, fillcolor="orange", opacity=0.2, line=dict(width=0)
        )
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=self.position.x - radius,
            y0=self.position.y - radius,
            x1=self.position.x + radius,
            y1=self.position.y + radius,
            fillcolor="orange",
            line=dict(width=0),
        )

    def get_agent_location(self):
        x = self.position.x
        z = self.position.y
        rotation = self.get_rotation()
        return {"x": x, "z": z, "rotation": rotation}


@AGENT_REGISTRY.register()
class TeachAgentReact(TeachAgent):
    def parse_response(self, response):
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.max_view_distance / self.step_size,
                        degree=self.aov / 2,
                        max_inventory=self.inventory_capacity,
                        max_manipulation_steps=self.max_manipulate_distance
                        / self.step_size,
                    ),
                )
            else:
                self.status["started"] = True
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "Start",
                        task_description=self.task_description,
                        observation=self.env.get_observation(self),
                    ),
                )
        else:
            thought = []
            act = []
            for line in response.split("\n"):
                line = line.strip()
                if len(line) == 0:
                    continue
                if any(
                    [
                        think_tok in line.lower()
                        for think_tok in ["thought:", "think:", "i think:"]
                    ]
                ):
                    thought.append("".join(line.split(":")[1:]))
                elif any([act_tok in line.lower() for act_tok in ["act:", "action:"]]):
                    act.append("".join(line.split(":")[1:]))
                elif len(act) == 0 and len(thought) > 0:
                    thought.append(line)
                else:
                    act.append(line)
            response = []
            if len(thought) > 0:
                response.append(("thought", " ".join(thought)))

            if len(act) > 0:
                act = " ".join(act).strip()
                act_resp = super().parse_response(act)
                response.append(("act", act))
                return dict(
                    success=act_resp.get("success", True),
                    response=response,
                    action=act_resp.get("action", "Pass"),
                    action_arg=act_resp.get("action_arg", {}),
                    feedback=act_resp.get("feedback", None),
                )
            return dict(response=response, feedback="OK.", success=True)
