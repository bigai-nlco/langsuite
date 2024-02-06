# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from gymnasium import spaces

from langsuite.agents import Agent
from langsuite.envs.base_env import ENV_REGISTRY, LangSuiteEnv
from langsuite.shapes import Line2D, Point2D
from langsuite.utils.logging import logger
from langsuite.world import World

ProcTHOR2DAction = {
    "MOVE_AHEAD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "PICK": 4,
    "PLACE": 5,
}


@ENV_REGISTRY.register()
class Iqa2DEnv(LangSuiteEnv):
    """Iqa environment class

    This class provides functions to:
        - Load scenes, agents.
        - Apply agent actions and perform simulation steps.

    Args:
        config (dict): Environment config
    """

    def __init__(self, env_config):
        super().__init__(env_config=env_config)
        self.agents = defaultdict()
        self.agent_names = list()
        self.current_status = {
            "object_in_view": [],
            "door_in_view": [],
            "wall_in_view": [],
            "room": "",
            "direction": "",
            "action_status": "",
            "message": "",
        }
        self.object_id2name = {}
        self.object_name2id = {}
        self.feedback_builder = None
        self.action_spaces = spaces.Discrete(len(ProcTHOR2DAction))
        self._history = dict()
        self.question = None
        self.answer = None
        self.question_type = None
        self.question_info = {}
        self.parent2children = {}
        self.children2parent = {}
        self.id2objects = {}
        self.count_number = 0

        self._terminated = False
        self.is_high = env_config.get('is_high_level')

    @property
    def is_multi_agent(self):
        """
            Determines if there are multiple agents.

            Returns:
                bool: True if the number of agents is greater than 1, False otherwise.
        """
        return len(self.agents) > 1

    @property
    def is_terminated(self):
        """
            Checks if the action is terminated.

            Returns:
                bool: True if the action is terminated, False otherwise.
        """
        return self._terminated

    def add_agent(self, agent_cfg) -> None:
        """
            Add an agent to the environment.

            Args:
                agent_cfg (dict): Configuration for the agent, including its attributes and parameters.
        """

        if "position" in agent_cfg:
            if isinstance(agent_cfg["position"], list):
                pass
            elif agent_cfg.get("position").lower() == "random":
                position = self.random_world_position()
                agent_cfg.update({"position": position})
        logger.info(agent_cfg)
        agent = Agent.create(deepcopy(agent_cfg))
        agent.max_view_distance = agent_cfg["max_view_distance"]
        agent.max_manipulate_distance = agent_cfg["max_manipulate_distance"]
        agent.view_degree = agent_cfg["view_degree"]
        agent.set_env(self)
        self.agents[agent.id] = agent
        self.agent_ids.append(agent.id)
        self._history[agent.id] = {"obs": [], "reward": [], "done": [], "info": []}

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

        for obj_json in world_cfg["data"]["objects"]:
            parentReceptacles = obj_json["parentReceptacles"]
            if parentReceptacles is not None:
                for p in parentReceptacles:
                    if p in parent2children:
                        parent2children[p].append(obj_json["objectId"])
                    else:
                        parent2children[p] = [obj_json["objectId"]]
        for obj_json in world_cfg["data"]["objects"]:
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
            self.id2objects[obj_id] = obj
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
                    self.id2objects[child.id] = child
                    obj_category_dic[child_category].append(child.id)

            obj_category_dic[category].append(obj_id)

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

    def step_single_agent(self, *, agent_id, action):
        if agent_id in self.agent_ids:
            success, info = self.agents[agent_id].step(action)
            info.update({"agent": agent_id})
            return None, 0, success, info

        logger.info(f"Agent {agent_id} not found in environment.")

        return None, None, 0, {}

    def reset(self):
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""

    def render(self, mode = "", **kwargs):
        """Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        figure = self.render_plotly()
        if mode == "webui":
            return figure
        return figure

    def close(self):
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""

    def get_task_def(self):
        """Get the question"""
        return self.question

    def get_answer(self):
        """Get the answer"""
        return self.answer

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

    def get_object_by_id(self, target_id):
        """
            Retrieve an object by its unique identifier.

            Args:
                target_id: The unique identifier of the object to be retrieved.

            Returns:
                object or None: The object with the specified ID, or None if not found.
        """
        for id, obj in self.world.objects.items():
            if id == target_id:
                return obj
            else:
                children = self.find_all_children(obj)
                for child in children:
                    if child.id == target_id:
                        return child
        return None

    @property
    def prev_obs(self):
        pass

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
                if (
                    obj.props["parentReceptacles"]
                    and "Floor" not in obj.props["parentReceptacles"][0]
                ):
                    parent_id = obj.props["parentReceptacles"][0]
                    parent_obj = self.world.objects[parent_id]
                    if parent_obj.props["openable"] and parent_obj.props["openness"]:
                        objs[id] = obj
                    elif not parent_obj.props["openable"]:
                        objs[id] = obj
                else:
                    objs[id] = obj
                    if (
                        "openable" in obj.props
                        and obj.props["openable"]
                        and not obj.props["isOpen"]
                    ):
                        continue
                    else:
                        if len(obj.children) > 0:
                            for child in obj.children.values():
                                objs[id] = child
        return objs

    def get_openned_object_observation(self, object_id):
        """
            Generate an observation for objects contained within an opened object.

            Args:
                object_id: The unique identifier of the opened object.

            Returns:
                str: An observation describing objects contained within the opened object.
        """
        children = []
        observation = "In/on it you see "
        if object_id in self.world.objects:
            obj = self.world.objects.get(object_id)
            for child in obj.children.keys():
                children.append(self.object_id2name[child])
        if len(children) > 0:
            observation += ", a ".join(children)
        else:
            observation += "nothing"
        return observation

    def get_look_around_observation(self, agent):
        observed_objects = self.get_receptacle_objects()
        observation = "You are in the middle of a room. Looking quickly around you, you see "
        
        if len(observed_objects) == 0:
            observation += "nothing."
        else:
            for obj_id in observed_objects:
                observation += (
                    "a " + self.object_id2name[obj_id] + ", "
                )
        if observation.strip().endswith(", "):
            observation = observation[:-1] + "."
        return observation

    def get_obj_description(self, obj):
        children = []
        description = ""
        obj_type = self.object_id2name[obj.id]
        temperature = ""
        is_dirty = ""
        if "temperature" in obj.props and obj.props["temperature"] == "Cold":
            temperature = "cool"
        elif "temperature" in obj.props and obj.props["temperature"] == "Hot":
            temperature = "hot"
        if "isDirty" in obj.props and obj.props["isDirty"]:
            is_dirty = "dirty"
        elif "isDirty" in obj.props and not obj.props["isDirty"]:
            is_dirty = "clean"
        if "receptacle" in obj.props and obj.props["receptacle"]:
            if obj.props["openable"] and not obj.props["isOpen"]:
                description = obj_type + " is closed. You can check it by opening it."
            elif len(obj.children) > 0:
                for child in obj.children.keys():
                    children.append(self.object_id2name[child])
                description = "In it, you see a " + ", a ".join(children) + "."
            elif len(obj.children) == 0:
                description = "In it, you see nothing."
        else:
            if temperature != "" or is_dirty != "":
                if temperature == "":
                    description = obj_type + " is " + is_dirty + "."
                elif is_dirty == "":
                    description = obj_type + " is " + temperature + "."
                else:
                    description = obj_type + " is " + temperature + " and " + is_dirty + "."
            elif "toggleable" in obj.props and obj.props["isToggled"]:
                description = obj_type + " is on."
            elif "toggleable" in obj.props and not obj.props["isToggled"]:
                description = obj_type + " is off."
            elif "sliceable" in obj.props and obj.props["sliceable"]:
                description = "This is a sliceable " + obj_type + "."
            else:
                description = "There's nothing special about " + obj_type + "."
        return description

    def get_receptacle_objects(self):
        objs = {}
        for id, obj in self.world.objects.items():
            if (
                "receptacle" in obj.props
                and obj.props["receptacle"]
            ):
                objs[id] = obj
        return objs

    def get_observation(self, agent, on_start = False):
        """
            Generate an observation based on the agent's field of view.

            Args:
                agent: The agent for which to generate the observation.

            Returns:
                str: An observation describing (Calculate lines of sight (middle, left, and right) 
                based on the agent's view vector.) objects within the agent's field of view.
        """
        if self.is_high and on_start:
            return self.get_look_around_observation(agent)

        observed_objects = self.get_observed_objects(agent)
        middle_objs = []
        left_objs = []
        right_objs = []
        middle_point = agent.position + agent.view_vector * agent.max_view_distance
        middle_line = Line2D([agent.position, middle_point])
        left_line = deepcopy(middle_line)
        left_line.rotate(-agent.aov / 2)
        right_line = deepcopy(middle_line)
        right_line.rotate(agent.aov / 2)

        for id, obj in observed_objects.items():
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
                middle_observation += "a " + self.object_id2name[obj.id] + "; "
                children = []
                if (
                    "openable" in obj.props
                    and obj.props["openable"]
                    and not obj.props["openness"]
                ):
                    continue
                else:
                    if len(obj.children) > 0:
                        for child in obj.children.keys():
                            children.append(self.object_id2name[child])
                if len(children) > 0:
                    middle_observation += ",".join(children) + " in/on it. "

        if len(left_objs) == 0:
            left_observation = ""
        else:
            left_observation = "On your left, you see "
            for obj in left_objs:
                left_observation += "a " + self.object_id2name[obj.id] + "; "
                children = []
                if (
                    "openable" in obj.props
                    and obj.props["openable"]
                    and not obj.props["openness"]
                ):
                    continue
                else:
                    if len(obj.children) > 0:
                        for child in obj.children.keys():
                            children.append(self.object_id2name[child])
                if len(children) > 0:
                    left_observation += ",".join(children) + " in/on it."

        if len(right_objs) == 0:
            right_observation = ""
        else:
            right_observation = "On your right, you see "
            for obj in right_objs:
                right_observation += "a " + self.object_id2name[obj.id] + ", "
                children = []
                if (
                    "openable" in obj.props
                    and obj.props["openable"]
                    and not obj.props["openness"]
                ):
                    continue
                else:
                    if len(obj.children) > 0:
                        for child in obj.children.keys():
                            children.append(self.object_id2name[child])
                if len(children) > 0:
                    right_observation += ", a ".join(children) + " in/on it. "
        observation = middle_observation + left_observation + right_observation
        if len(observation) == 0:
            if (self.is_high):
                observation = "You see nothing. You can goto other objects to explore the room."
            else:
                observation = "You see nothing. You can try to take action like move_ahead, turn_left or turn_right to explore the room."
        return observation

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
        # fig.show()
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
            # logger.debug(objid)
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
