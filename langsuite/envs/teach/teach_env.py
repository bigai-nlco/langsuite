# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from langsuite.agents import Agent
from langsuite.envs.base_env import ENV_REGISTRY, LangSuiteEnv
from langsuite.shapes import Line2D, Point2D
from langsuite.utils.logging import logger
from langsuite.world import World

Teach2DAction = {
    "MOVE_AHEAD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "PICK": 4,
    "PLACE": 5,
}

TeachPath = Path(__file__).parent


@ENV_REGISTRY.register()
class Teach2DEnv(LangSuiteEnv):
    """Teach environment class

    This class provides functions to:
        - Load scenes, agents.
        - Apply agent actions and perform simulation steps.

    Args:
        config (dict): Environment config
    """

    def __init__(self, env_config):
        super().__init__(env_config=env_config)
        self.agents = defaultdict()
        self.agent_ids = list()
        self.agent_names = list()
        self.feedback_builder = None
        self.object_id2name = {}
        self.object_name2id = {}
        self.parent2children = {}
        self.children2parent = {}
        self.expert_steps = 0
        self.obj_category_dic = {}

        self.action_id2name = {
            0: "Stop",
            1: "Move to",
            2: "Forward",
            3: "Backward",
            4: "Turn Left",
            5: "Turn Right",
            6: "Look Up",
            7: "Look Down",
            8: "Pan Left",
            9: "Pan Right",
            10: "Move Up",
            11: "Move Down",
            12: "Double Forward",
            13: "Double Backward",
            300: "Navigation",
            200: "Pickup",
            201: "Place",
            202: "Open",
            203: "Close",
            204: "ToggleOn",
            205: "ToggleOff",
            206: "Slice",
            207: "Dirty",
            208: "Clean",
            209: "Fill",
            210: "Empty",
            211: "Pour",
            212: "Break",
            400: "BehindAboveOn",
            401: "BehindAboveOff",
            500: "OpenProgressCheck",
            501: "SelectOid",
            502: "SearchObject",
            100: "Text",
            101: "Speech",
            102: "Beep",
        }

    @property
    def is_multi_agent(self):
        """
        Determines if there are multiple agents.

        Returns:
            bool: True if the number of agents is greater than 1, False otherwise.
        """
        return len(self.agents) > 1

    def set_feedback_builder(self, feedback_builder):
        self.feedback_builder = feedback_builder

    def random_world_position(self):
        """
        Generate a random world position.

        Returns:
            list: A list containing X and Y coordinates of a random position in the environment.
        """
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
        """
        Create the world based on a given configuration.

        Args:
            world_cfg (dict): Configuration data for the world, including object relationships.
        """
        parent2children = {}
        init_state = world_cfg["data"]["tasks"][0]["episodes"][0]["initial_state"]
        for obj_json in init_state["objects"]:
            parentReceptacles = obj_json["parentReceptacles"]
            if parentReceptacles is not None:
                for p in parentReceptacles:
                    if p in parent2children:
                        parent2children[p].append(obj_json["objectId"])
                    else:
                        parent2children[p] = [obj_json["objectId"]]
        for obj_json in init_state["objects"]:
            if obj_json["objectId"] in parent2children:
                obj_json.update(
                    {"receptacleObjectIds": parent2children[obj_json["objectId"]]}
                )
        self.world = World.create(world_cfg)

        obj_category_dic = defaultdict(list)
        for obj_id, obj in self.world.objects.items():
            category = obj_id.split("|")[0]
            if category not in obj_category_dic:
                obj_index = 0
                obj_category_dic[category] = []
            else:
                obj_index = len(obj_category_dic[category])
            name = category + "_" + str(obj_index)
            name = name.lower()
            self.object_id2name[obj_id] = name
            self.object_name2id[name] = obj_id
            children = self.find_all_children(obj)
            if len(children) > 0:
                for child in children:
                    child_category = child.id.split("|")[0]
                    if category not in obj_category_dic:
                        child_index = 0
                        obj_category_dic[child_category] = []
                    else:
                        child_index = len(obj_category_dic[child_category])
                    child_name = child_category + "_" + str(child_index)
                    child_name = child_name.lower()
                    self.object_id2name[child.id] = child_name
                    self.object_name2id[child_name] = child.id
                    obj_category_dic[child_category].append(child.id)

            obj_category_dic[category].append(obj_id)
        self.obj_category_dic = obj_category_dic

    def find_all_children(self, obj):
        """
        Recursively find all children of a given object.

        Args:
            obj: The object for which to find all children.

        Returns:
            list: A list of all child objects, including their descendants.
        """
        children = []
        if len(obj.children) > 0:
            for child in obj.children.values():
                children.append(child)
                children.extend(self.find_all_children(child))
        return children

    def add_agent(self, agent_cfg) -> None:
        """
        Add an agent to the environment.

        Args:
            agent_cfg (dict): Configuration for the agent, including its attributes and parameters.
        """

        if "position" in agent_cfg:
            if isinstance(agent_cfg["position"], list) or isinstance(
                agent_cfg["position"], Point2D
            ):
                agent_cfg.update({"position": agent_cfg["position"]})
            elif agent_cfg.get("position").lower() == "random":
                position = self.random_world_position()
                agent_cfg.update({"position": position})
        logger.info(agent_cfg)
        agent = Agent.create(deepcopy(agent_cfg))
        agent.set_env(self)
        self.agents[agent.agent_id] = agent
        self.agent_ids.append(agent.agent_id)
        self.agent_names.append(agent.agent_name)
        self._history[agent.agent_id] = {
            "obs": [],
            "reward": [],
            "done": [],
            "info": [],
        }

    def is_valid_action(self, action: str) -> bool:
        return True

    def update_config(self, config):
        """
        Update the configuration of agents in the environment.

        Args:
            config (dict): New configuration data for agents, where each agent's configuration is specified.
        """
        for i, agent_id in enumerate(self.agents):
            self.agents[agent_id].set_config(config["agents"][i])

    def step_single_agent(self, agent_id: str, action):
        """Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info."""
        status, info = self.agents[agent_id].step(action)
        info["agent"] = agent_id
        return None, 0, status, info

    def expert_step(self, action_dict):
        list_obs, list_info = {}, {}
        list_obs["n"] = [None, None]
        list_info["n"] = [None, None]

        while self.expert_steps < len(self.interactions):
            interaction = self.interactions[self.expert_steps]
            if self.action_id2name[interaction["action_id"]] not in [
                "Navigation",
                "Look Up",
                "Look Down",
            ]:
                break
            self.expert_steps += 1
            interaction = self.interactions[self.expert_steps]
        if self.expert_steps >= len(self.interactions):
            info = {"is_terminated": True}
            return list_obs, None, None, list_info
        agent_id = interaction["agent_id"]
        action_id = interaction["action_id"]
        action_dict["action"] = self.action_id2name[action_id]
        action_dict["action_arg"] = interaction
        obs, _, _, info = self.step_single_agent(agent_id=agent_id, action=action_dict)
        list_obs["n"][agent_id] = obs
        list_info["n"][agent_id] = info
        if self.expert_steps >= len(self.interactions):
            info = {"is_terminated": True}
            return list_obs, None, None, list_info
        return list_obs, None, None, list_info

    def reset(self):
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""
        for agent_id in self.agent_ids:
            self.agents[agent_id].reset()

    def render(self, mode="", **kwargs):
        """Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        figure = self.render_plotly()
        # figure = self.render_matplotlib()
        if mode == "webui":
            return figure
        return figure

    def close(self):
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""

    def get_feedback_builder(self):
        return self.feedback_builder

    def render_plotly(self):
        """
        Render the virtual environment using Plotly for visualization.

        Returns:
            return the Plotly figure for visualization.
        """
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
        fig.show()
        return fig

    def render_matplotlib(self, save_to_path=None):
        """
        Render the virtual environment using Matplotlib for visualization.

        Returns:
            return the Matplotlib figure for visualization.
        """
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
            obj.plot(axes=axes)

        for _, agent in self.agents.items():
            agent.plot(axes=axes)

        if save_to_path is not None:
            plt.savefig(save_to_path)

        plt.show()

    def is_valid_trajectory(self, traj):
        """
        Check if a trajectory is valid and collision-free.

        Args:
            traj (Point2D or Line2D): The trajectory to be checked.

        Returns:
            bool: True if the trajectory is collision-free, False if it encounters obstacles.
        """
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
        """
        Determine the room where an agent is located.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            Room or None: The room where the agent is located, or None if not found.
        """
        for room_id, room in self.rooms.items():
            if room.geometry.contains(self.agents[agent_id].position):
                return room
        return None

    def get_agent_position(self, agent_id: str):
        """
        Get the position and rotation of the specified agent.

        Args:
            agent_id (str): The unique identifier of the agent.

        Returns:
            dict: A dictionary containing agent position (x, z) and rotation.
        """
        x = self.agents[agent_id].position.x
        z = self.agents[agent_id].position.y
        rotation = self.agents[agent_id].rotation
        return {"x": x, "z": z, "rotation": rotation}

    def get_objects_in_exposed_obj(self, obj):
        """Iteratively find all unobstructed objects"""
        objs = {}
        objs[obj.id] = obj
        if not (
            "openable" in obj.props
            and obj.props["openable"]
            and not obj.props["isOpen"]
        ):
            for child in obj.children.values():
                objs.update(self.get_objects_in_exposed_obj(child))
        return objs

    def get_observed_objects(self, agent):
        """
        Retrieve objects that the agent can observe based on its position.

        Args:
            agent: The agent whose observation capability is considered.

        Returns:
            dict: A dictionary of observed objects with their unique IDs as keys.
        """
        objs = {}
        for id, obj in self.world.objects.items():
            if agent.can_observe(obj.geometry):
                objs.update(self.get_objects_in_exposed_obj(obj))
        return objs

    def get_clearly_observed_objects(self, agent):
        """Returning objects within the field of view radius"""
        objs = {}
        for id, obj in self.world.objects.items():
            if agent.can_observe(obj.geometry):
                objs[id] = obj
        return objs

    def get_observation_in_object(self, obj):
        """
        Generate an observation for objects contained in or on the object.

        Args:
            object_id: The unique identifier of the object.

        Returns:
            str: An observation describing objects contained in or on the object.
        """
        observation = ""
        if (
            "openable" in obj.props
            and obj.props["openable"]
            and not obj.props["isOpen"]
        ):
            observation += "a closed " + self.object_id2name[obj.id]
        elif "openable" in obj.props and obj.props["openable"] and obj.props["isOpen"]:
            observation += "an opened " + self.object_id2name[obj.id]
            observation += ", "

            if len(obj.children) > 0:
                children_observation = []
                for child in obj.children.values():
                    children_observation += [self.get_observation_in_object(child)]
                observation += "there is " + "".join(children_observation) + "in it"
            else:
                observation += "it's empty"
        else:
            observation += "a " + self.object_id2name[obj.id]
            observation += ", "

            if len(obj.children) > 0:
                children_observation = []
                for child in obj.children.values():
                    children_observation += [self.get_observation_in_object(child)]
                observation += "there is " + "".join(children_observation) + "on it"
            else:
                observation = observation.rstrip(", ")
        observation += ", "
        return observation

    def get_openned_object_observation(self, object_id):
        """
        Generate an observation for objects contained within an opened object.

        Args:
            object_id: The unique identifier of the opened object.

        Returns:
            str: An observation describing objects contained within the opened object.
        """
        children = []
        observation = "In it you see "
        if object_id in self.world.objects:
            obj = self.world.objects.get(object_id)
            for child in obj.children.keys():
                children.append(self.object_id2name[child])
        if len(children) > 0:
            observation += ", a ".join(children)
        else:
            observation += "nothing"
        return observation

    def get_held_object_observation(self, agent):
        """
        Describing the things in hand
        """
        inventory = []
        observation = "You are now holding "

        for obj in agent.inventory:
            inventory.append(self.object_id2name[obj.id])
        if len(inventory) > 0:
            observation += (
                ", a ".join(inventory) + ". Drop one before you pick up another one. "
            )
        else:
            observation += "nothing. "
        return observation

    def get_observation(self, agent):
        """
        Generate an observation based on the agent's field of view.

        Args:
            agent: The agent for which to generate the observation.

        Returns:
            str: An observation describing (Calculate lines of sight (middle, left, and right)
            based on the agent's view vector.) objects within the agent's field of view.
        """
        middle_objs = []
        left_objs = []
        right_objs = []
        middle_point = agent.position + agent.view_vector * agent.max_view_distance
        middle_line = Line2D([agent.position, middle_point])
        left_line = deepcopy(middle_line)
        left_line.rotate(-agent.aov / 2)
        right_line = deepcopy(middle_line)
        right_line.rotate(agent.aov / 2)
        observed_objects = self.get_clearly_observed_objects(agent)
        oppo_agent_observation = ""
        oppo_agent = self.agents[agent.opponent_agent_id]

        if agent.can_observe(oppo_agent.position):
            min_distance = float("inf")
            idx = -1
            for i, line in enumerate([middle_line, left_line, right_line]):
                oppo_agent_line_distance = oppo_agent.position.shapely_geo.distance(
                    line.shapely_geo
                )
                if oppo_agent_line_distance < min_distance:
                    min_distance = oppo_agent_line_distance
                    idx = i
            if idx == 0:
                oppo_agent_observation += f"{oppo_agent.agent_name} is in front of you."
            elif idx == 1:
                oppo_agent_observation += f"{oppo_agent.agent_name} is on your left."
            elif idx == 2:
                oppo_agent_observation += f"{oppo_agent.agent_name} is on your right."
            else:
                raise ValueError("Invalid index.")

        for _, obj in observed_objects.items():
            distance_dict = {
                "middle_distance": obj.geometry.shapely_geo.distance(
                    middle_line.shapely_geo
                ),
                "left_distance": obj.geometry.shapely_geo.distance(
                    left_line.shapely_geo
                ),
                "right_distance": obj.geometry.shapely_geo.distance(
                    right_line.shapely_geo
                ),
            }

            min_dis = sorted(distance_dict.items(), key=lambda dis: dis[1])
            if min_dis[0][0] == "middle_distance":
                middle_objs.append(obj)
            elif min_dis[0][0] == "left_distance":
                left_objs.append(obj)
            elif min_dis[0][0] == "right_distance":
                right_objs.append(obj)

        if len(middle_objs) == 0:
            middle_observation = ""
        else:
            middle_observation = "In front of you, You see "
            for obj in middle_objs:
                middle_observation += self.get_observation_in_object(obj)
                middle_observation = middle_observation.rstrip(", ") + "; "

        if len(left_objs) == 0:
            left_observation = ""
        else:
            left_observation = "On your left, you see "
            for obj in left_objs:
                left_observation += self.get_observation_in_object(obj)
                left_observation = left_observation.rstrip(", ") + "; "
        if len(right_objs) == 0:
            right_observation = ""
        else:
            right_observation = "On your right, you see "
            for obj in right_objs:
                right_observation += self.get_observation_in_object(obj)
                right_observation = right_observation.rstrip(", ") + "; "
        if len(middle_observation) > 0:
            middle_observation = middle_observation.strip("; ")
            middle_observation += ". "
        if len(left_observation) > 0:
            left_observation = left_observation.strip("; ")
            left_observation += ". "
        if len(right_observation) > 0:
            right_observation = right_observation.strip("; ")
            right_observation += ". "
        observation = middle_observation + left_observation + right_observation
        if len(observation) == 0:
            observation = "You see nothing. You can try to take action like move_ahead, turn_left or turn_right to explore the room."

        return observation
