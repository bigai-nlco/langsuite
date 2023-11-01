# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import math
import re
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY, Agent
from langsuite.shapes import Cone2D, Point2D, Polygon2D, Vector2D
from langsuite.utils import math_utils
from langsuite.utils.logging import logger


class MessageParser(object):
    def __init__(self, mode=None) -> None:
        self.mode = mode or "regex"

    def parse(self, message):
        if self.mode == "regex":
            res = re.findall(r"[\w]+\(\w.*\)$", message.strip())

            actions = []
            if res:
                logger.debug(res)
                for a in res:
                    logger.debug(a)
                    a = a.strip().split("(")
                    action_name = a[0].strip()
                    arg_str = "".join(a[1:]).split(")")
                    arg_str = "".join(arg_str[:-1]).split(",")
                    # args = args.split(')')
                    args = []
                    kwargs = {}
                    for arg in arg_str:
                        arg = arg.strip()
                        if "=" in arg:
                            k, v = arg.split("=")
                            kwargs[k.strip()] = v.strip()
                        else:
                            args.append(arg)
                    actions.append(
                        {"action": action_name, "args": args, "kwargs": kwargs}
                    )

            return actions


@AGENT_REGISTRY.register()
class SimpleAgent(Agent):
    def __init__(self, agent_config: Dict) -> None:
        super().__init__(agent_config=agent_config)

        self.view_vector = Vector2D(0, 1)
        self.position = Point2D(agent_config.get("position"))
        self.set_config(agent_config)
        self.view_geometry = self._compute_fov_geometry()
        # logger.info(self.view_geometry)
        self.message_parser = MessageParser()

        self.inventory = []
        logger.info(f"Successfully add agent: {self.cfg}")

    @classmethod
    def create(cls, agent_cfg: Dict):
        return cls(agent_config=agent_cfg)

    def step(self, action_dict):
        if type(action_dict) == str:
            action_dict = dict(action=action_dict)

        prompt = action_dict.get("prompt")
        action = action_dict.get("action")
        if not action and prompt and len(prompt) > 0:
            action = self.parse_prompt(prompt)
        action_status = self.execute(action=action)
        return action_status.success, dict(feedback=action_status.feedback)

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

    def parse_prompt(self, message=None):
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
        # left_view_point = self.position + self.view_vector * self.max_view_distance
        # left_view_point.rotate(angle=-(self.aov / 2), center=self.position)
        # right_view_point = self.position + self.view_vector * self.max_view_distance
        # right_view_point.rotate(angle=(self.aov / 2), center=self.position)
        return Cone2D(
            center=self.position,
            radius=self.max_view_distance,
            direction=self.view_vector,
            angle=self.aov,
        )

    # TODO  view obstables is not taken into consideration
    # TODO  it's not reasonable when the object is very big.
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
