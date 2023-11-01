# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import math
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from langsuite.agents import Agent
from langsuite.envs.base_env import ENV_REGISTRY, LangSuiteEnv
from langsuite.envs.cwah import utils
from langsuite.shapes import Line2D, Point2D
from langsuite.utils.logging import logger
from langsuite.world import World

Cwah2DAction = {"MOVE_AHEAD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3, "PICK": 4, "PLACE": 5}

CwahPath = Path(__file__).parent


@ENV_REGISTRY.register()
class Cwah2DEnv(LangSuiteEnv):
    def __init__(self, env_config):
        super().__init__(env_config=env_config)
        self.agents = defaultdict()
        self.agent_ids = list()
        self.agent_names = list()
        self.feedback_builder = None
        # self.action_spaces = spaces.Discrete(len(Teach2DAction))
        # self._history = dict()
        # self._terminated = False
        self.object_id2name = {}
        self.object_name2id = {}
        self.parent2children = {}
        self.children2parent = {}
        self.expert_steps = 0
        self.obj_category_dic = {}

        self.task_goal = env_config["task_goal"]
        self.goal_class = env_config["goal_class"]
        self.task_name = env_config["task_name"]
        self.full_graph = utils.inside_not_trans(env_config["full_graph"])
        self.num_agents = env_config["num_agents"]
        self.goal_spec = {
            agent_id: self.get_goal(self.task_goal[agent_id])
            for agent_id in range(self.num_agents)
        }

    @property
    def is_multi_agent(self):
        return len(self.agents) > 1

    # @property
    # def is_terminated(self):
    #     return self._terminated

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

    def random_agent_position_in_room(self, room_class):
        for room in self.world.rooms.values():
            if room.class_name == room_class:
                rand_room_poly = room.geometry
                break
        print(
            [
                [rand_room_poly.x_min, rand_room_poly.y_min],
                [rand_room_poly.x_max, rand_room_poly.y_max],
            ]
        )
        rand_position = Point2D(
            np.random.randint(
                [
                    math.ceil(rand_room_poly.x_min + 1),
                    math.ceil(rand_room_poly.y_min + 1),
                ],
                [math.floor(rand_room_poly.x_max), math.floor(rand_room_poly.y_max)],
            ).tolist()
        )
        if self.is_valid_trajectory(rand_position):
            logger.info(f"Found valid position: {rand_position}")
            print([rand_position.x, rand_position.y])
            return rand_position
        else:
            return self.random_agent_position_in_room(room_class)

    def generate_name(self, category, obj_id):
        # return "<" + category + "> " + "(" + str(obj_id) + ")"
        return category + "_" + str(obj_id)

    def create_world(self, world_cfg) -> None:
        self.world = World.create(world_cfg)
        obj_category_dic = defaultdict(list)
        for obj_id, obj in self.world.id2object.items():
            category = obj.class_name
            if category not in obj_category_dic:
                obj_index = 0
                obj_category_dic[category] = []
            else:
                obj_index = len(obj_category_dic[category])
            # obj_index = obj.id
            name = self.generate_name(category, obj_index)
            name = name.lower()
            self.object_id2name[obj_id] = name
            self.object_name2id[name] = obj_id
            obj_category_dic[category].append(obj_id)
        self.obj_category_dic = obj_category_dic

    def add_agent(self, agent_cfg) -> None:
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
        for i, agent_id in enumerate(self.agents):
            self.agents[agent_id].set_config(config["agents"][i])

    def step_single_agent(self, agent_id: str, action):
        """Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info."""
        status, info = self.agents[agent_id].step(action)
        # if "Stop" == info.get("action"):
        #     info["is_terminated"] = True
        info["agent"] = agent_id
        return None, 0, status, info

    def reset(self):
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""
        reset_dict = {}
        reset_dict["containers_name"] = self.all_containers_name
        reset_dict["goal_objects_name"] = self.all_goal_objects_name
        reset_dict["rooms_name"] = self.all_room_name
        reset_dict["room_info"] = self.room_info

        for agent_id in self.agents:
            agent = self.agents[agent_id]
            reset_dict["goal"] = self.get_goal(self.task_goal[agent_id])
            agent.reset(reset_dict)

    def render(self, mode="", **kwargs):
        """Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        figure = self.render_plotly()
        if mode == "webui":
            return figure
        return figure

    def close(self):
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""

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

    def get_agent_position(self, agent_id: str):
        x = self.agents[agent_id].position.x
        z = self.agents[agent_id].position.y
        rotation = self.agents[agent_id].rotation
        return {"x": x, "z": z, "rotation": rotation}

    def get_goal(self, task_spec):
        goal_class = {}
        for predicate in self.goal_class.keys():
            rel, obj1, obj2 = predicate.split("_")
            goal_class[f"{rel}_{obj1}"] = obj2
        new_task_goal = {}
        for predicate, count in task_spec.items():
            if count == 0:
                continue
            rel, obj1, obj2 = predicate.split("_")
            obj2_name = goal_class[f"{rel}_{obj1}"]
            new_predicate = predicate.replace(obj2, f"<{obj2_name}> ({obj2})")
            new_task_goal[new_predicate] = count
        # print(new_task_goal)
        res_dict = {
            goal_k: [goal_c, True, 2] for goal_k, goal_c in new_task_goal.items()
        }
        # res_dict.update(predicates_grab)
        return res_dict

    @property
    def all_relative_name(self) -> list:
        return (
            self.all_containers_name
            + self.all_goal_objects_name
            + self.all_room_name
            + ["character"]
        )

    @property
    def all_relative_id(self) -> list:
        return [
            node["id"]
            for node in self.full_graph["nodes"]
            if node["class_name"] in self.all_relative_name
        ]

    @property
    def all_detection_id(self) -> list:
        return [
            node["id"]
            for node in self.full_graph["nodes"]
            if node["class_name"] in self.detection_all_object
        ]

    @property
    def all_containers_name(self) -> list:
        r"""
        get all containers in the scene, exclude rooms and containers with no objects inside.
        """
        """
        id2node = {node['id']: node for node in self.full_graph['nodes']}
        room_name = [node['class_name'] for node in self.full_graph['nodes'] if node['category'] == 'Rooms']
        all_container = list(set([id2node[link['to_id']]['class_name'] for link in self.full_graph['edges'] if
                                  link['relation_type'] == 'INSIDE']))
        all_container = [x for x in all_container if x not in room_name]
        """
        container_classes = [
            "bathroomcabinet",
            "kitchencabinet",
            "cabinet",
            "fridge",
            "stove",
            # 'coffeepot',
            "dishwasher",
            "microwave",
        ]
        return container_classes

    @property
    def all_goal_objects_name(self) -> list:
        r"""
         get all objects that related to goal.
        ZHX: update to adapt to new goal_spec of LLM
        """
        goal_objects = []
        id2node = {node["id"]: node for node in self.full_graph["nodes"]}
        for predicate in self.goal_spec[0]:
            elements = predicate.split("_")
            for x in elements[1:]:
                if x.isdigit():
                    goal_objects += [id2node[int(x)]["class_name"]]
                elif "(" in x:
                    y = x.split("(")[1].split(")")[0]
                    if y.isdigit():
                        goal_objects += [id2node[int(y)]["class_name"]]
                else:
                    goal_objects += [x]
        goal_obj = list(set(goal_objects))
        # if ('character' not in goal_obj):
        # 	goal_obj += ['character']
        return goal_obj

    @property
    def room_info(self):
        r"""
        get room info in the scene.
        """
        return [
            node
            for node in self.full_graph["nodes"]
            if node["id"] in self.all_room_and_character_id
        ]

    @property
    def all_room_name(self) -> list:
        r"""
        get all rooms in the scene.
        """
        # room_name = [node['class_name'] for node in self.full_graph['nodes'] if node['category'] == 'Rooms']
        room_name = ["livingroom", "kitchen", "bedroom", "bathroom"]
        return room_name

    @property
    def all_room_and_character_id(self) -> list:
        r"""
        get all room_and_character_ids in the scene.
        """
        return [
            node["id"]
            for node in self.full_graph["nodes"]
            if node["class_name"] == "character" or node["category"] in ["Rooms"]
        ]

    @property
    def all_room_id(self) -> list:
        r"""
        get all room_and_character_ids in the scene.
        """
        return [
            node["id"]
            for node in self.full_graph["nodes"]
            if node["category"] in ["Rooms"]
        ]

    def get_openned_object_observation(self, object_id):
        children = []
        # observation = "In/on it you see "
        observation = "In it you see "
        for e in self.full_graph["edges"]:
            if e["relation_type"] == "INSIDE" and e["to_id"] == object_id:
                children.append(self.object_id2name[e["from_id"]])
        if len(children) > 0:
            observation += ", a ".join(children)
        else:
            observation += "nothing"
        return observation

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

    def get_observation_in_object(self, obj):
        observation = ""
        if "CAN_OPEN" in obj.props["properties"] and "CLOSED" in obj.props["states"]:
            observation += "a closed " + self.object_id2name[obj.id]
        elif "CAN_OPEN" in obj.props["properties"] and "OPEN" in obj.props["states"]:
            observation += "an opened " + self.object_id2name[obj.id]
            observation += ", "
            children_observation = []
            for e in self.env.full_graph["edges"]:
                if e["relation_type"] == "INSIDE" and e["to_id"] == obj.id:
                    children_observation += [
                        self.get_observation_in_object(
                            self.env.world.id2object[e["from_id"]]
                        )
                    ]
            if len(children_observation) > 0:
                observation += "there is " + "".join(children_observation) + "in/on it"
            else:
                observation += "it's empty"
        else:
            observation += "a " + self.object_id2name[obj.id]
        observation += ", "
        return observation

    def get_visible_nodes(self, graph, agent_id):
        # Obtains partial observation from the perspective of agent_id
        # That is, objects inside the same room as agent_id and not inside closed containers
        # NOTE: Assumption is that the graph has an inside transition that is not transitive
        state = graph
        id2node = {node["id"]: node for node in state["nodes"]}
        rooms_ids = [
            node["id"] for node in graph["nodes"] if node["category"] == "Rooms"
        ]

        character = id2node[agent_id]

        # find character
        character_id = character["id"]
        inside_of, is_inside, edge_from = {}, {}, {}

        grabbed_ids = []
        for edge in state["edges"]:
            if edge["relation_type"] == "INSIDE":
                if edge["to_id"] not in is_inside.keys():
                    is_inside[edge["to_id"]] = []

                is_inside[edge["to_id"]].append(edge["from_id"])
                inside_of[edge["from_id"]] = edge["to_id"]

            elif "HOLDS" in edge["relation_type"]:
                if edge["from_id"] == character["id"]:
                    grabbed_ids.append(edge["to_id"])

        if character_id not in inside_of.keys():
            print(inside_of)
        character_inside_ids = inside_of[character_id]
        room_id = character_inside_ids

        object_in_room_ids = is_inside[room_id]

        # Some object are not directly in room, but we want to add them
        curr_objects = list(object_in_room_ids)
        while len(curr_objects) > 0:
            objects_inside = []
            for curr_obj_id in curr_objects:
                new_inside = (
                    is_inside[curr_obj_id] if curr_obj_id in is_inside.keys() else []
                )
                objects_inside += new_inside

            object_in_room_ids += list(objects_inside)
            curr_objects = list(objects_inside)

        # Only objects that are inside the room and not inside something closed
        # TODO: this can be probably speed up if we can ensure that all objects are either closed or open
        object_hidden = (
            lambda ido: inside_of[ido] not in rooms_ids
            and "OPEN" not in id2node[inside_of[ido]]["states"]
        )
        observable_object_ids = [
            object_id
            for object_id in object_in_room_ids
            if not object_hidden(object_id)
        ] + rooms_ids
        observable_object_ids += grabbed_ids

        partilly_observable_state = {
            "edges": [
                edge
                for edge in state["edges"]
                if edge["from_id"] in observable_object_ids
                and edge["to_id"] in observable_object_ids
            ],
            "nodes": [id2node[id_node] for id_node in observable_object_ids],
        }

        return partilly_observable_state
