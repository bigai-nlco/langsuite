# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.agents.simple_agent import SimpleAgent
from langsuite.llms import create_llm, create_llm_prompts, process_llm_results
from langsuite.llms.output_parsers import RegexOutputParser
from langsuite.shapes import Point2D, Vector2D
from langsuite.utils.logging import logger
from langsuite.utils.math_utils import euclidean_distance


@AGENT_REGISTRY.register()
class CWAHAgent(SimpleAgent):
    """
    Cwah agent class
    """

    def __init__(self, agent_config: Dict) -> None:
        super().__init__(agent_config=agent_config)
        # self.isExpert = agent_config.get("isExpert", False)
        self.agent_id = agent_config["agent_id"]
        self.opponent_agent_id = 1 - self.agent_id
        self.agent_name = "Alice" if self.agent_id == 0 else "Bob"
        self.oppo_name = "Alice" if self.agent_id == 1 else "Bob"
        self.oppo_pronoun = "she" if self.agent_id == 1 else "he"

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

        self.rooms_name = []
        self.roomname2id = {}
        self.room_info = None
        self.id2node = {}
        self.id_inside_room = {}
        self.reachable_objects = []
        self.goal = dict(
            goal_objects_name=[],
            goal_location=None,
            goal_location_id=None,
        )
        self.state = dict(
            stuck=False, containers_name=None, current_room=None, last_room=None
        )
        self.satisfied = []
        self.unsatisfied = {}
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }

    @classmethod
    def create(cls, agent_cfg: Dict):
        return cls(agent_config=agent_cfg)

    def step(self, action_dict):
        """
        :param action_dict: The action dictionary"""

        if type(action_dict) == str:
            action_dict = dict(prompt=action_dict)

        self._update()
        prompt = action_dict.get("prompt")
        self.current_prompt = prompt
        parsed_response = {}
        response = self.fetch_prompt_response(prompt)
        parsed_response = self.parse_response(response)
        logger.info(parsed_response)
        success = parsed_response.get("success", True)
        if (
            success
            and "action" in parsed_response
            and parsed_response["action"] != "UserInput"
            and parsed_response["action"] != "Pass"
        ):
            if parsed_response["action"] == "Stop":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "Stop", "success"
                )
                # parsed_response["is_terminated"] = True
                success = True
            else:
                print(parsed_response["action"], parsed_response["action_arg"])
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
            # {"role": "assistant", "content": response_content},
        ]
        # else:
        #     self.chat_history[0]["content"] = self.get_progress()
        #     self.chat_history += [
        #         {"role": "assistant", "content": response},
        #     ]
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

    def _get_obj_id_by_name(self, name):
        """Get object id by name"""
        name = name.lower()
        if name in self.rooms_name:
            for room_id, room in self.env.world.rooms.items():
                if name in room.class_name:
                    return room_id

        # possible_objects = []
        objname_patern = re.compile(r"[a-zA-Z]+_[0-9]")
        match = objname_patern.match(name)
        if match and name in self.env.object_name2id:
            target_id = self.env.object_name2id[name]
            return target_id
        return None

    def _justify_action_name(self, name):
        name = name.lower()
        splits = name.split("_")
        return "".join([s[0].upper() + s[1:] for s in splits])

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
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "Start",
                        task_description=self.goal_desc,
                        observation=self.get_progress(),
                    ),
                )
        else:
            if "]" in response[0][1]:
                response[0][1] = response[0][1].split("]")[0] + "]"
            if ":" in response[0][1]:
                response[0][1] = response[0][1].split(":")[1]
            parsed_actions = self.output_parser.parse(response[0][1].lower())

            action_dicts = []
            action_names = [
                "chat",
                "go_explore",
                "go_check",
                "go_grab",
                "go_put",
            ]
            for action_name, action_arg in parsed_actions:
                obj_name = action_arg.strip("'").strip('"')
                receptacle_name = None
                if "," in obj_name:
                    obj_name = action_arg.split(",")[0].strip("'").strip('"').strip()
                    receptacle_name = (
                        action_arg.split(",")[1].strip("'").strip('"').strip()
                    )
                # chat, stop generate [message] not [object_id], go_put has fixed args [grabbed_object, goal_location]
                if action_name in ["chat", "stop", "go_put"]:
                    action_dicts.append(
                        dict(
                            action_name=self._justify_action_name(action_name),
                            action_arg={"chat_response": obj_name},
                        )
                    )
                    continue

                obj_id = self._get_obj_id_by_name(obj_name)

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
                receptacle_id = None
                # if receptacle_name:
                #     receptacle_id = self._get_obj_id_by_name(receptacle_name)
                #     if receptacle_id is None:
                #         return dict(
                #             success=False,
                #             response=response,
                #             feedback=self.env.feedback_builder.build(
                #                 "InvalidAction",
                #                 "failure.objectNotProvide",
                #                 object=receptacle_name,
                #             ),
                #         )
                #     logger.info("receptacle name: {} receptacle id {}.".format(receptacle_name, receptacle_id))

                action_dicts.append(
                    dict(
                        action_name=self._justify_action_name(action_name),
                        action_arg={
                            "object_id": obj_id,
                            "object_name": obj_name,
                            "receptacle_id": receptacle_id,
                            "receptacle_name": receptacle_name,
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

    def is_close(self, agent, node):
        """
        Check if the node is close to agent
        """
        agent_position = Point2D(
            agent["obj_transform"]["position"][0], agent["obj_transform"]["position"][2]
        )
        node_position = Point2D(
            node["obj_transform"]["position"][0], node["obj_transform"]["position"][2]
        )
        return euclidean_distance(agent_position, node_position) < 1.5

    def check_progress(self, state, goal_spec):
        unsatisfied = {}
        satisfied = []
        id2node = {node["id"]: node for node in state["nodes"]}
        total_cnt = 0
        for key, value in goal_spec.items():
            elements = key.split("_")
            cnt = value[0]
            total_cnt += cnt
            for edge in state["edges"]:
                if cnt == 0:
                    break
                if (
                    edge["relation_type"].lower() == elements[0]
                    and edge["to_id"] == self.goal["goal_location_id"]
                    and id2node[edge["from_id"]]["class_name"] == elements[1]
                ):
                    satisfied.append(id2node[edge["from_id"]])
                    cnt -= 1
                    # if self.debug:
                    # 	print(satisfied)
            if cnt > 0:
                unsatisfied[key] = cnt
        statistics = len(satisfied) / total_cnt
        return satisfied, unsatisfied, statistics

    def reset(self, reset_dict):
        """
        Reset current scene graph and dialogue agent
        """
        obs = self.env.full_graph

        self.view_vector = Vector2D(0, 1)

        self.set_config(self.init_cfg)

        self.steps = 0

        self.state["containers_name"] = reset_dict.get("containers_name")
        self.goal["goal_objects_name"] = reset_dict.get("goal_objects_name")
        self.rooms_name = reset_dict.get("rooms_name")
        self.roomname2id = {x["class_name"]: x["id"] for x in reset_dict["room_info"]}

        self.state["stuck"] = False
        self.state["last_room"] = None
        self.unsatisfied = {k: v[0] for k, v in reset_dict["goal"].items()}

        self.satisfied.clear()
        self.inventory.clear()
        # self.dialogue_history.clear()

        self.goal["goal_location"] = list(reset_dict["goal"].keys())[0].split("_")[-1]
        self.goal["goal_location_id"] = int(
            self.goal["goal_location"].split(" ")[-1][1:-1]
        )
        self.id_inside_room = {
            self.goal["goal_location_id"]: deepcopy(self.rooms_name),
            self.opponent_agent_id: None,
        }
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.opponent_grabbed_objects = []

        # self.action_history = [
        #     f"goexplore [{self.current_room['class_name']}> ({self.current_room['id']}]"
        # ]
        # self.dialogue_history = []
        agent_node = {
            "id": self.agent_id,
            "category": "Characters",
            "class_name": "character",
            "obj_transform": {
                "position": [self.position.x, 0, self.position.y],
                "rotation": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
            "states": [],
            "properties": [],
        }
        for node in obs["nodes"]:
            agent2node = {}
            agent2node["from_id"] = self.agent_id
            agent2node["to_id"] = node["id"]
            if ("GRABBABLE" in node["properties"]) or (
                "CAN_OPEN" in node["properties"]
            ):
                if self.is_close(agent_node, node):
                    agent2node["relation_type"] = "CLOSE"
                else:
                    agent2node["relation_type"] = "FAR"
                obs["edges"].append(agent2node)
        obs["nodes"].append(agent_node)

        self.id2node = {x["id"]: x for x in obs["nodes"]}
        for room_id, room in self.env.world.rooms.items():
            if room.geometry.contains(self.position):
                agent_in_room = {}
                agent_in_room["from_id"] = self.agent_id
                agent_in_room["to_id"] = room_id
                agent_in_room["relation_type"] = "INSIDE"
                obs["edges"].append(agent_in_room)
                self.current_room = self.id2node[room_id]

        self.goal_desc, self.goal_location_with_r = self.goal2description(
            self.unsatisfied, None
        )

    def _update(self):
        """
        Modified from https://github.com/UMass-Foundation-Model/Co-LLM-Agents/blob/master/envs/cwah/agents/LLM_agent.py#L181
        Get agent current observation and status
        """
        agent_node = self.id2node[self.agent_id]
        agent_node["obj_transform"]["position"] = [self.position.x, 0, self.position.y]
        goal = self.env.get_goal(self.env.task_goal[self.agent_id])

        for room_id, room in self.env.world.rooms.items():
            if room.geometry.contains(self.position):
                for e in self.env.full_graph["edges"]:
                    if e["from_id"] == self.agent_id and e["relation_type"] == "INSIDE":
                        e["to_id"] = room_id
                self.current_room = self.id2node[room_id]
                break

        obs = self.env.get_visible_nodes(self.env.full_graph, self.agent_id)

        satisfied, unsatisfied, statistic = self.check_progress(
            self.env.full_graph, goal
        )
        # print(f"satisfied: {satisfied}")
        # if len(satisfied) > 0:
        self.unsatisfied = unsatisfied
        self.satisfied = satisfied
        self.goal_desc, self.goal_location_with_r = self.goal2description(
            self.unsatisfied, None
        )

        obs = self._filter_graph(obs)

        for e in obs["edges"]:
            node = self.id2node[e["to_id"]]
            if (
                e["from_id"] == self.agent_id
                and "HOLD" not in e["relation_type"]
                and (
                    ("GRABBABLE" in node["properties"])
                    or ("CAN_OPEN" in node["properties"])
                )
            ):
                if self.is_close(agent_node, node):
                    e["relation_type"] = "CLOSE"
                else:
                    e["relation_type"] = "FAR"

        self.grabbed_objects = []
        opponent_grabbed_objects = []
        self.reachable_objects = []

        for e in obs["edges"]:
            x, r, y = e["from_id"], e["relation_type"], e["to_id"]
            if x == self.agent_id:
                if r in ["HOLDS_RH", "HOLDS_LH"]:
                    self.grabbed_objects.append(y)
                    nodey = self.id2node[y]
                    nodey["obj_transform"]["position"] = [
                        self.position.x,
                        0,
                        self.position.y,
                    ]

                elif r == "CLOSE":
                    y = self.id2node[y]
                    # self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
                    self.reachable_objects.append(self.env.object_id2name[y["id"]])
            elif x == self.opponent_agent_id and r in ["HOLDS_RH", "HOLDS_LH"]:
                opponent_grabbed_objects.append(self.id2node[y])

        unchecked_containers = []
        ungrabbed_objects = []

        for x in obs["nodes"]:
            if x in self.grabbed_objects or x in self.opponent_grabbed_objects:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None:
                        continue
                    j = None
                    for i, ungrab in enumerate(ungrabbed):
                        if x == ungrab:
                            j = i
                    if j is not None:
                        ungrabbed.pop(j)
                continue
            self.id_inside_room[x["id"]] = self.current_room["class_name"]
            if (
                x["class_name"] in self.state["containers_name"]
                and "CLOSED" in x["states"]
                and x["id"] != self.goal["goal_location_id"]
            ):
                unchecked_containers.append(x)
            if (
                any([x["class_name"] == g.split("_")[1] for g in self.unsatisfied])
                and all([x["id"] != y["id"] for y in self.satisfied])
                and "GRABBABLE" in x["properties"]
                and x["id"] not in self.grabbed_objects
                and x["id"] not in [w["id"] for w in opponent_grabbed_objects]
            ):
                ungrabbed_objects.append(x)

        if (
            type(self.id_inside_room[self.goal["goal_location_id"]]) is list
            and self.current_room["class_name"]
            in self.id_inside_room[self.goal["goal_location_id"]]
        ):
            self.id_inside_room[self.goal["goal_location_id"]].remove(
                self.current_room["class_name"]
            )
            if len(self.id_inside_room[self.goal["goal_location_id"]]) == 1:
                self.id_inside_room[
                    self.goal["goal_location_id"]
                ] = self.id_inside_room[self.goal["goal_location_id"]][0]
        self.unchecked_containers[
            self.current_room["class_name"]
        ] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room["class_name"]] = ungrabbed_objects[:]

        info = {
            "graph": obs,
            "obs": {
                "grabbed_objects": self.grabbed_objects,
                "opponent_grabbed_objects": self.opponent_grabbed_objects,
                "reachable_objects": self.reachable_objects,
                "progress": {
                    "unchecked_containers": self.unchecked_containers,
                    "ungrabbed_objects": self.ungrabbed_objects,
                },
                "satisfied": self.satisfied,
                "current_room": self.current_room["class_name"],
            },
        }
        if (
            self.id_inside_room[self.opponent_agent_id]
            == self.current_room["class_name"]
        ):
            self.opponent_grabbed_objects = opponent_grabbed_objects

        return obs, info

    def _filter_graph(self, obs):
        relative_id = [
            node["id"]
            for node in obs["nodes"]
            if node["class_name"] in self.env.all_relative_name
        ]

        relative_id = [
            x for x in relative_id if all([x != y["id"] for y in self.satisfied])
        ]

        new_graph = {
            "edges": [
                edge
                for edge in obs["edges"]
                if edge["from_id"] in relative_id and edge["to_id"] in relative_id
            ],
            "nodes": [node for node in obs["nodes"] if node["id"] in relative_id],
        }
        return new_graph

    def goal2description(self, goals, goal_location_room):  # {predicate: count}
        # print(goals)
        map_rel_to_pred = {
            "inside": "into",
            "on": "onto",
        }
        s = "Find and put "
        r = None
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split("_")
            count = vl
            if count == 0:
                continue
            if relation == "holds":
                continue
                # s += f"Alice holds a book, "
            elif relation == "sit":
                continue
                # s += f"Alice sits in {obj2}, "
            else:
                s += f"{count} {obj1}{'s' if count > 1 else ''}, "
                r = relation
        if r is None:
            return (
                "None.",
                f"to the {self.env.object_id2name[self.goal['goal_location_id']]}",
            )

        s = (
            s[:-2]
            + f" {map_rel_to_pred[r]} the {self.env.object_id2name[self.goal['goal_location_id']]}."
        )
        # if type(goal_location_room) is not list:
        # 	s += f" in the {goal_location_room}."
        # else:
        # 	ss = ' or '.join([f'{room}' for room in goal_location_room])
        # 	s += f", which may be in the {ss}."
        return (
            s,
            f"{map_rel_to_pred[r]} the {self.env.object_id2name[self.goal['goal_location_id']]}",
        )

    def get_progress(self):
        return self.progress2text(
            self.current_room,
            [self.id2node[go] for go in self.grabbed_objects],
            self.unchecked_containers,
            self.ungrabbed_objects,
            self.id_inside_room[self.goal["goal_location_id"]],
            self.satisfied,
            self.opponent_grabbed_objects,
            self.id_inside_room[self.opponent_agent_id],
            room_explored=None,
        )

    def progress2text(
        self,
        current_room,
        grabbed_objects,
        unchecked_containers,
        ungrabbed_objects,
        goal_location_room,
        satisfied,
        opponent_grabbed_objects,
        opponent_last_room,
        room_explored,
    ):
        sss = {}
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                # extra_obj = self.goal["goal_location"]
                extra_obj = self.env.object_id2name[self.goal["goal_location_id"]]
            if (
                objs is None
                and extra_obj is None
                and (room_explored is None or not room_explored[room])
            ):
                sss[room] = f"The {room} is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"{extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += self.env.object_id2name[x["id"]]
                else:
                    ss = ", ".join([self.env.object_id2name[x["id"]] for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"an unchecked container {self.env.object_id2name[x['id']]}"
                else:
                    ss = ", ".join([self.env.object_id2name[x["id"]] for x in cons])
                    s_con = "unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += "nothing"
                if room_explored is not None and not room_explored[room]:
                    s += " yet"
            elif s_obj != "" and s_con != "":
                s += s_obj + ", and " + s_con
            else:
                s += s_obj + s_con
            sss[room] = s

        if len(satisfied) == 0:
            s = ""
        else:
            s = "You have already found and put "
            s += ", ".join([self.env.object_id2name[x["id"]] for x in satisfied])
            s += " " + self.goal_location_with_r + ". "
        # s += self.goal_desc
        if len(grabbed_objects) == 0:
            s += "You are holding nothing. "
        else:
            s += (
                f"You are holding {self.env.object_id2name[grabbed_objects[0]['id']]}. "
            )
            if len(grabbed_objects) == 2:
                s = (
                    s[:-2]
                    + f" and {self.env.object_id2name[grabbed_objects[1]['id']]}. "
                )
        s += f"You are in the {current_room['class_name']}, where you found {sss[current_room['class_name']]}. "
        ### opponent modeling
        # if not self.single:
        ss = ""
        if len(opponent_grabbed_objects) == 0:
            ss += "nothing. "
        else:
            ss += f"{self.env.object_id2name[opponent_grabbed_objects[0]['id']]}. "
            if len(opponent_grabbed_objects) == 2:
                ss = (
                    ss[:-2]
                    + f" and {self.env.object_id2name[opponent_grabbed_objects[0]['id']]}). "
                )
        if opponent_last_room is None:
            s += f"You don't know where {self.oppo_name} is. "
        elif opponent_last_room == current_room["class_name"]:
            s += f"You also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
        else:
            s += f"Last time you saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms_name:
            if room == current_room["class_name"]:
                continue
            if "unexplored" in sss[room]:
                s += sss[room]
            else:
                s += f"You found {sss[room]} in the {room}. "

        return s


@AGENT_REGISTRY.register()
class CWAHAgentReact(CWAHAgent):
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
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "Start",
                        # task_description=self.env.goal_spec[self.agent_id],
                        task_description=self.goal_desc,
                        observation=self.get_progress(),
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
            return dict(
                response=response,
                action="Pass",
                action_arg={"chat_response": ""},
                feedback="OK.",
                success=True,
            )
