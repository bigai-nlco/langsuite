# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from langsuite.agents import Agent
from langsuite.envs.base_env import ENV_REGISTRY, LangSuiteEnv
from langsuite.shapes import Line2D, Point2D
from langsuite.utils.logging import logger
from langsuite.world import World

Rearrange2DAction = {
    "MOVE_AHEAD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "PICK": 4,
    "PLACE": 5,
}


@ENV_REGISTRY.register()
class Rearrange2DEnv(LangSuiteEnv):
    def __init__(self, env_config):
        super().__init__(env_config=env_config)
        self.agents = defaultdict()
        # self.feedback_builder = TemplateBuilder(env_config.get("template"))
        # self.action_spaces = spaces.Discrete(len(Rearrange2DAction))
        self.closed = False
        self.object_id2name = {}
        self.object_name2id = {}
        self.parent2children = {}
        self.children2parent = {}
        self.target_pose_description = None
        # self.id2objects = {}

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
        parent2children = {}
        children2parent = {}
        for obj_json in world_cfg["data"]["objects"]:
            parentReceptacles = obj_json["parentReceptacles"]
            children2parent[obj_json["objectId"]] = []
            if parentReceptacles is not None:
                for p in parentReceptacles:
                    if "Floor|" in p:
                        continue
                    children2parent[obj_json["objectId"]].append(p)
                    if p in parent2children:
                        parent2children[p].append(obj_json["objectId"])
                    else:
                        parent2children[p] = [obj_json["objectId"]]
        for obj_json in world_cfg["data"]["objects"]:
            if obj_json["objectId"] in parent2children:
                if obj_json["objectId"] in children2parent:
                    c2p = children2parent[obj_json["objectId"]]
                    p2c = parent2children[obj_json["objectId"]]

                    if not set(c2p).isdisjoint(set(p2c)):
                        obj_json.update(
                            {"parentReceptacles": list(set(c2p) - set(p2c))}
                        )
                        obj_json.update(
                            {"receptacleObjectIds": list(set(p2c) - set(c2p))}
                        )
                    else:
                        obj_json.update({"receptacleObjectIds": p2c})

        self.parent2children = parent2children
        self.children2parent = children2parent
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
            # self.id2objects[obj_id] = obj
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
                    # self.id2objects[child.id] = child
                    obj_category_dic[child_category].append(child.id)

            obj_category_dic[category].append(obj_id)

            self.obj_category_dic = obj_category_dic

    def find_all_children(self, obj):
        children = []
        if len(obj.children) > 0:
            for child in obj.children.values():
                children.append(child)
                children.extend(self.find_all_children(child))
        return children

    def add_agent(self, agent_cfg) -> None:
        # raise
        if len(self.agents) == 0:
            if "position" in agent_cfg:
                p = agent_cfg.get("position")
                if isinstance(p, str) and p.lower() == "random":
                    position = self.random_world_position()
                    agent_cfg.update({"position": position})
                    logger.info(agent_cfg)
            agent = Agent.create(agent_cfg)
            agent.set_env(self)
            if "name" in agent_cfg:
                agent.set_name(agent_cfg["name"])
            else:
                agent.set_name("Rearrange Agent")
            agent.update()
            self.agents[agent.id] = agent
            self.agent_ids.append(agent.id)
        else:
            logger.error("Rearrangement only need one agent!")
            raise

    def is_valid_action(self, action: str) -> bool:
        return True

    def update_config(self, config):
        for i, agent_id in enumerate(self.agents):
            self.agents[agent_id].update_config(config["agents"][i])

    # def get_task_info(self):

    #     return {
    #         "state": ActionFeedback(
    #             success=True, feedback=self.feedback_builder.build("intro")
    #         ).to_json()
    #     }

    def step(self, action_dict):
        """Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info."""
        if len(self.agents.keys()) > 0:
            for id in self.agents.keys():
                agent_id = id
                break
            status, parsed_response = self.agents[agent_id].step(action_dict)
            if "Stop" == parsed_response.get("action"):
                info = {
                    "status": status,
                    "agent": agent_id,
                    "feedback": parsed_response["feedback"],
                    "action": parsed_response.get("action"),
                    "response": parsed_response.get("response"),
                    "is_terminated": True,
                }
            else:
                info = {
                    "status": status,
                    "agent": agent_id,
                    "feedback": parsed_response["feedback"],
                    "action": parsed_response.get("action"),
                    "response": parsed_response.get("response"),
                }
        else:
            logger.info(f"Agent {agent_id} not found in environment.")
            info = {
                "status": False,
                "agent": agent_id,
                "feedback": f"Agent {agent_id} not found in environment.",
            }
        obs = None
        reward = 0
        return obs, reward, False, info

    def reset(self):
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""

    def render(self, mode="", **kwargs):
        """Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        figure = self.render_plotly()
        if mode == "webui":
            return figure
        return figure

    def set_feedback_builder(self, feedback_builder):
        self.feedback_builder = feedback_builder

    def close(self):
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""

    def get_task_def(self):
        # TODO
        return "rearrange"

    def get_target_status(self):
        # TODO
        return "rearrange"

    def get_feedback_builder(self):
        return self.feedback_builder

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
        fig.show()
        return fig

    def get_object_by_id(self, target_id):
        for id, obj in self.world.objects.items():
            if id == target_id:
                return obj
            else:
                children = self.find_all_children(obj)
                for child in children:
                    if child.id == target_id:
                        return child
        return None

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
                                # TODO there may be other obstacles
                                return True
                    return False

            for _, obj in self.world.objects.items():
                # TODO
                if not obj.id.startswith("Floor") and obj.geometry.intersects(traj):
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

    def get_agent_position(self, agent_id: str):
        x = self.agents[agent_id].position.x
        z = self.agents[agent_id].position.y
        rotation = self.agents[agent_id].rotation
        return {"x": x, "z": z, "rotation": rotation}

    def get_observed_objects(self, agent):
        objs = {}
        for id, obj in self.world.objects.items():
            if agent.can_observe(obj.geometry):
                objs[id] = obj
                if (
                    "openable" in obj.props
                    and obj.props["openable"]
                    and not obj.props["isOpen"]
                ):
                    continue
                else:
                    children = self.find_all_children(obj)
                    if len(children) > 0:
                        for child in children:
                            objs[child.id] = child
        return objs

    def reset_world(self, world_data):
        self.object_id2name.clear()
        self.object_name2id.clear()

        self.create_world(world_data)

    def get_openned_object_observation(self, object_id):
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

    def get_room_description(self):
        """get target pose description and then reset the envioriment"""
        room_observation = ""
        rand_room_id = list(self.world.rooms.keys())[0]
        rand_room_poly = self.world.rooms[rand_room_id].geometry
        pisition_objects_map = {
            "north": [],
            "northeast": [],
            "east": [],
            "southeast": [],
            "south": [],
            "southwest": [],
            "west": [],
            "northwest": [],
            "middle": [],
        }
        width_grid_size = (rand_room_poly.x_max - rand_room_poly.x_min) / 3
        length_grid_size = (rand_room_poly.y_max - rand_room_poly.y_min) / 3
        for obj in self.world.objects.values():
            x = floor((obj.position.x - rand_room_poly.x_min) / width_grid_size)
            y = floor((obj.position.y - rand_room_poly.y_min) / length_grid_size)
            if (x, y) == (0, 0):
                pisition_objects_map["southwest"].append(obj)
            elif (x, y) == (0, 1):
                pisition_objects_map["west"].append(obj)
            elif (x, y) == (0, 2):
                pisition_objects_map["northwest"].append(obj)
            elif (x, y) == (1, 0):
                pisition_objects_map["south"].append(obj)
            elif (x, y) == (1, 1):
                pisition_objects_map["middle"].append(obj)
            elif (x, y) == (1, 2):
                pisition_objects_map["north"].append(obj)
            elif (x, y) == (2, 0):
                pisition_objects_map["southeast"].append(obj)
            elif (x, y) == (2, 1):
                pisition_objects_map["east"].append(obj)
            elif (x, y) == (2, 2):
                pisition_objects_map["northeast"].append(obj)
        for location, objs in pisition_objects_map.items():
            observation = ""
            if len(objs) != 0:
                observation = "In the " + location + " of the room, there is "
            for obj in objs:
                if "openable" in obj.props and obj.props["openable"]:
                    if obj.props["isOpen"]:
                        observation += (
                            "an " + "opened " + self.object_id2name[obj.id] + "; "
                        )
                    else:
                        observation += "a " + self.object_id2name[obj.id] + "; "
                else:
                    observation += "a " + self.object_id2name[obj.id] + "; "
                children = []
                if len(obj.children) > 0:
                    for child in obj.children.keys():
                        children.append(self.object_id2name[child])
                if len(children) > 0:
                    observation = observation.strip("; ")
                    observation += ", "
                    observation += (
                        "in/on it you can see a " + ", a ".join(children) + "; "
                    )
            if len(observation) > 0:
                observation = observation.strip("; ")
                observation += ". "
            room_observation += observation
        return room_observation

    def get_held_object_observation(self, agent):
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

    def get_agent_position_observation(self, agent):
        agent_description = "Now you are in the {} of the room facing {}. "
        rand_room_id = list(self.world.rooms.keys())[0]
        rand_room_poly = self.world.rooms[rand_room_id].geometry
        width_grid_size = (rand_room_poly.x_max - rand_room_poly.x_min) / 3
        length_grid_size = (rand_room_poly.y_max - rand_room_poly.y_min) / 3
        width_grid_size = (rand_room_poly.x_max - rand_room_poly.x_min) / 3
        length_grid_size = (rand_room_poly.y_max - rand_room_poly.y_min) / 3
        x = floor((agent.position.x - rand_room_poly.x_min) / width_grid_size)
        y = floor((agent.position.y - rand_room_poly.y_min) / length_grid_size)
        position = None
        direction = None
        if (x, y) == (0, 0):
            position = "southwest"
        elif (x, y) == (0, 1):
            position = "west"
        elif (x, y) == (0, 2):
            position = "northwest"
        elif (x, y) == (1, 0):
            position = "south"
        elif (x, y) == (1, 1):
            position = "middle"
        elif (x, y) == (1, 2):
            position = "north"
        elif (x, y) == (2, 0):
            position = "southeast"
        elif (x, y) == (2, 1):
            position = "east"
        elif (x, y) == (2, 2):
            position = "northeast"

        if (agent.view_vector.x, agent.view_vector.y) == (0, 1):
            direction = "north"
        elif (agent.view_vector.x, agent.view_vector.y) == (1, 0):
            direction = "east"
        elif (agent.view_vector.x, agent.view_vector.y) == (0, -1):
            direction = "south"
        elif (agent.view_vector.x, agent.view_vector.y) == (-1, 0):
            direction = "west"

        if not position or not direction:
            agent_description = ""

        return agent_description.format(position, direction)

    def get_observation(self, agent):
        # self.render_matplotlib()
        observed_objects = self.get_observed_objects(agent)
        large_objs = []
        middle_objs = []
        left_objs = []
        right_objs = []
        middle_point = agent.position + agent.view_vector * agent.max_view_distance
        middle_line = Line2D([agent.position, middle_point])
        left_line = deepcopy(middle_line)
        left_line.rotate(-agent.aov / 2)
        right_line = deepcopy(middle_line)
        right_line.rotate(agent.aov / 2)
        # objects_sorted_by_area = sorted(
        #     observed_objects.items(), key=lambda obj: obj[1].geometry.area
        # )
        # for i in range(3):
        #     if i < len(objects_sorted_by_area):
        #         obj = objects_sorted_by_area[i]
        #         if obj[1].geometry.area > 0.1:
        #             large_objs.append(deepcopy(obj[1]))
        #             del observed_objects[obj[0]]
        #     else:
        #         break

        # for obj in large_objs:
        #     if (
        #         "openable" in obj.props
        #         and obj.props["openable"]
        #         and not obj.props["isOpen"]
        #     ):
        #         continue
        #     else:
        #         if len(obj.children) > 0:
        #             for child in obj.children.values():
        #                 del observed_objects[child.id]

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

        # if len(large_objs) == 0:
        #     large_observation = ""
        # else:
        #     large_observation = "Firstly you see "
        #     for obj in large_objs:
        #         large_observation += "a/an " + obj.id.split("|")[0] + "; "
        #         children = []
        #         if (
        #             "openable" in obj.props
        #             and obj.props["openable"]
        #             and not obj.props["isOpen"]
        #         ):
        #             continue
        #         else:
        #             if len(obj.children) > 0:
        #                 for child in obj.children.keys():
        #                     children.append(child.split("|")[0])
        #         if len(children) > 0:
        #             large_observation += ",".join(children) + " in/on it."
        if len(middle_objs) == 0:
            middle_observation = ""
        else:
            described_object = []
            middle_observation = "In front of you, You see "
            for obj in middle_objs:
                children = []
                if obj.id in described_object:
                    continue
                described_object.append(obj.id)
                if (
                    "openable" in obj.props
                    and obj.props["openable"]
                    and obj.props["isOpen"]
                ):
                    if len(obj.children) > 0:
                        for child in obj.children.keys():
                            described_object.append(child)
                            children.append(self.object_id2name[child])

                    if len(children) > 0:
                        middle_observation += (
                            "an opened " + self.object_id2name[obj.id] + ", "
                        )
                        middle_observation += (
                            "there is a " + ", a ".join(children) + " in/on it; "
                        )
                    else:
                        middle_observation += (
                            "an opened " + self.object_id2name[obj.id] + ", "
                        )
                        middle_observation += "it's empty; "
                else:
                    middle_observation += "a " + self.object_id2name[obj.id] + "; "

        if len(left_objs) == 0:
            left_observation = ""
        else:
            described_object = []
            left_observation = "On your left, you see "
            for obj in left_objs:
                if obj.id in described_object:
                    continue
                children = []
                described_object.append(obj.id)
                if (
                    "openable" in obj.props
                    and obj.props["openable"]
                    and obj.props["isOpen"]
                ):
                    if len(obj.children) > 0:
                        for child in obj.children.keys():
                            described_object.append(child)
                            children.append(self.object_id2name[child])

                    if len(children) > 0:
                        left_observation += (
                            "an opened " + self.object_id2name[obj.id] + ", "
                        )
                        left_observation += (
                            "there is a " + ", a ".join(children) + " in/on it; "
                        )
                    else:
                        left_observation += (
                            "an opened" + self.object_id2name[obj.id] + ", "
                        )
                        left_observation += "it's empty; "
                else:
                    left_observation += "a " + self.object_id2name[obj.id] + "; "

        if len(right_objs) == 0:
            right_observation = ""
        else:
            described_object = []
            right_observation = "On your right, you see "
            for obj in right_objs:
                if obj.id in described_object:
                    continue
                described_object.append(obj.id)
                children = []
                if (
                    "openable" in obj.props
                    and obj.props["openable"]
                    and obj.props["isOpen"]
                ):
                    if len(obj.children) > 0:
                        for child in obj.children.keys():
                            described_object.append(child)
                            children.append(self.object_id2name[child])

                    if len(children) > 0:
                        right_observation += (
                            "an opened " + self.object_id2name[obj.id] + ", "
                        )
                        right_observation += (
                            "there is a " + ", a ".join(children) + " in/on it; "
                        )
                    else:
                        right_observation += (
                            "an opened " + self.object_id2name[obj.id] + ", "
                        )
                        right_observation += "it's empty; "
                else:
                    right_observation += "a " + self.object_id2name[obj.id] + "; "

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

        # agent_position_observation = self.get_agent_position_observation(agent)
        return observation

    def get_all_object_pos(self):
        all_poses = {}
        for obj_id, obj in self.world.objects.items():
            pos = {"obj_id": obj_id, "obj_position": (obj.position.x, obj.position.x)}
            if "openable" in obj.props and obj.props["openable"]:
                pos["openable"] = True
                if obj.props["isOpen"]:
                    pos["isOpen"] = True
                else:
                    pos["isOpen"] = False
            else:
                pos["openable"] = False
                pos["isOpen"] = None
            all_poses[obj_id] = pos
            children = obj.find_all_children()
            for child in children:
                c_pos = {
                    "obj_id": child.id,
                    "obj_position": (child.position.x, child.position.x),
                }
                if "openable" in child.props and child.props["openable"]:
                    c_pos["openable"] = True
                    if obj.props["isOpen"]:
                        c_pos["isOpen"] = True
                    else:
                        c_pos["isOpen"] = False
                else:
                    c_pos["openable"] = False
                    c_pos["isOpen"] = None
                all_poses[child.id] = c_pos
        return all_poses

    def get_diff_count(self, starts, targets):
        diff_count = 0
        for start_obj, target_obj in zip(starts, targets):
            assert start_obj["name"] == target_obj["name"]
            if start_obj["position"] != target_obj["position"]:
                diff_count += 1
            if "openness" in target_obj:
                diff_count += 1
        return diff_count
