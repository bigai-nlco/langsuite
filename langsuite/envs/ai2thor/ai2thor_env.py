# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from gymnasium import spaces

from langsuite.actions import ActionFeedback
from langsuite.agents import Agent
from langsuite.envs.base_env import ENV_REGISTRY, LangSuiteEnv
from langsuite.shapes import Line2D, Point2D
from langsuite.utils.logging import logger
from langsuite.world import World

AI2THOR2DAction = {
    "MOVE_AHEAD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "PICK": 4,
    "PLACE": 5,
}


@ENV_REGISTRY.register()
class AI2ThorEnv(LangSuiteEnv):
    def __init__(self, env_config):
        super().__init__(env_config=env_config)
        self.agents = defaultdict()
        self.agent_ids = list()
        self.current_status = {
            "object_in_view": [],
            "door_in_view": [],
            "wall_in_view": [],
            "room": "",
            "direction": "",
            "action_status": "",
            "message": "",
        }
        self.feedback_builder = None
        self.action_spaces = spaces.Discrete(len(AI2THOR2DAction))
        self._history = dict()

        self._terminated = False

    @property
    def is_multi_agent(self):
        return len(self.agents) > 1

    @property
    def is_terminated(self):
        return self._terminated

    def set_feedback_builder(self, feedback_builder):
        self.feedback_builder = feedback_builder

    def random_world_position(self):
        if self.world:
            rand_room_id = random.choice(list(self.world.rooms.keys()))
            rand_room_poly = self.world.rooms[rand_room_id].geometry
            rand_position = Point2D(
                np.random.randint(
                    [rand_room_poly.x_min, rand_room_poly.y_min],
                    [rand_room_poly.x_max + 1, rand_room_poly.y_max + 1],
                ).tolist()
            )
            if self.is_valid_trajectory(rand_position):
                logger.info(f"Found valid position: {rand_position}")
                return [rand_position.x, rand_position.y]
            else:
                return self.random_world_position()

        raise RuntimeError("World is not initialized.")

    def create_world(self, world_cfg) -> None:
        self.world = World.create(world_cfg)

    # def create_grid_map(self,):

    def is_valid_action(self, action: str) -> bool:
        return True

    def update_config(self, config):
        for i, agent_id in enumerate(self.agents):
            self.agents[agent_id].set_config(config["agents"][i])

    def step_single_agent(self, *, agent_id, action):
        if isinstance(agent_id, Agent):
            agent_id = agent_id.id

        if agent_id in self.agent_ids:
            success, info = self.agents[agent_id].step(action)
            info.update({"agent": agent_id})
            return None, 0, success, info

        logger.info(f"Agent {agent_id} not found in environment.")

        return None, None, 0, {}

    def reset(self):
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""

    def render(self, mode="", **kwargs):
        """Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        figure = self.render_plotly()
        if mode == "webui":
            return figure
        return figure

    def close(self):
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""

    @property
    def prev_obs(self):
        pass

    def get_feedback_builder(self):
        return self.feedback_builder

    def get_task_info(self):
        return {
            "state": ActionFeedback(
                success=True, feedback=self.feedback_builder.build("intro")
            )
        }

    def render_plotly(self):
        fig = go.Figure()

        for _, room in self.world.rooms.items():
            room.render(fig)

        for _, door in self.world.doors.items():
            door.render(fig)

        for _, obj in self.world.objects.items():
            # logger.debug(objid)
            obj.render(fig)

        for _, agent in self.agents.items():
            agent.render(fig)

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(showlegend=False)
        # fig.show()
        return fig

    def render_matplotlib(self, save_to_path=None):
        fig = plt.figure(num=3, figsize=(5, 5))
        axes = fig.add_subplot(1, 1, 1)
        for _, room in self.world.rooms.items():
            room.plot(axes=axes)
        for _, wall in self.world.walls.items():
            wall.plot(axes=axes)
        for _, door in self.world.doors.items():
            door.plot(axes=axes)
        for _, window in self.world.windows.items():
            window.plot(axes=axes)
        for objid, obj in self.world.objects.items():
            # logger.debug(objid)
            obj.plot(axes=axes)

        for _, agent in self.agents.items():
            agent.plot(axes=axes)

        if save_to_path is not None:
            plt.savefig(save_to_path)

        plt.show()

    def is_valid_trajectory(self, traj):
        if isinstance(traj, Point2D):
            traj = Line2D([traj, Point2D(traj.x + 1, traj.y + 1)])
        elif not isinstance(traj, Line2D):
            raise ValueError(
                f"'traj' has to be of type Point2D | Line2D ({type(traj)} given)"
            )

        if len(traj.coords) < 2:
            return True

        if len(traj.coords) == 2:
            for _, wall in self.world.walls.items():
                if wall.geometry.intersects(traj):
                    if len(wall.doors) > 0:
                        for _, door in wall.doors.items():
                            if door.geometry.intersects(traj) and door.is_open:
                                return True
                    return False

            for _, obj in self.world.objects.items():
                if obj.geometry.intersects(traj):
                    return False
            return True
        else:
            for i, coord in enumerate(traj.coords[:-1]):
                segment = Line2D([coord, traj[i + 1]])

                if not self.is_valid_trajectory(segment):
                    return False
            return True

    def locate_agent_room(self, agent_id: str):
        for room_id, room in self.rooms.items():
            if room.geometry.contains(self.agents[agent_id].position):
                return room
        # logger.info("cannot locate agent room!")
        return None
