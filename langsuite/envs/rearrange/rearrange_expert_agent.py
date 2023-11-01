# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import math
from copy import deepcopy

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.envs.rearrange.rearrange_agent import RearrangeAgent
from langsuite.shapes import Point2D
from langsuite.utils.grid_world import GridWorld, cal_wall_min_max, get_direction
from langsuite.utils.logging import logger


@AGENT_REGISTRY.register()
class RearrangeExpertAgent(RearrangeAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config=agent_config)
        self.status = dict(started=False)
        self.chat_history = []
        self.target_status = agent_config.get("target_status")
        self.start_status = agent_config.get("start_status")
        self.plans = self.generate_plan(self.start_status, self.target_status)
        self.actions = ["YES."]

    def query_expert(self):
        if len(self.actions) == 0:
            if len(self.plans) == 0:
                return "stop"
            plan = self.plans.pop(0)
            object_id = plan["object_id"]
            # self.env.render()
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
                children = obj.find_all_children()
                for child in children:
                    child_x_list = list(child.geometry.shapely_geo.exterior.xy[0][:4])
                    child_y_list = list(child.geometry.shapely_geo.exterior.xy[1][:4])

                    grid_world.add_object(
                        child.position.x, child.position.y, (child_x_list, child_y_list)
                    )
                    # grid_world.render()
            if "Drop" in plan["action"]:
                target_position = (plan["position"]["x"], plan["position"]["y"])
            else:
                object = self.env.world.get_object(object_id)

                if not object:
                    self.actions.clear()
                    self.plans.clear()
                    return "Stop"
                target_position = (object.position.x, object.position.y)
            (
                end_coordinate,
                action_trajectory,
                opetation_rotate_list,
            ) = grid_world.get_path(
                self.position,
                target_position,
                agent_direction,
                grid_world.grid,
            )

            # if action_trajectory is None:
            #     raise
            # self.env.render()
            self.actions.extend(action_trajectory)
            # for action in action_trajectory:
            #     self.execute(action=action)
            dest = Point2D(end_coordinate[0], end_coordinate[1])
            # self.set_position(dest)
            # for rotate_action in opetation_rotate_list:
            #     if rotate_action == "TurnRight":
            #         self.agent.rotate(self.degree)
            #     elif rotate_action == "TurnLeft":
            #         self.agent.rotate(-self.degree)

            if plan["action"] == "PickUp":
                if (
                    object_id in self.env.children2parent
                    and len(self.env.children2parent[object_id]) > 0
                ):
                    parent_ids = self.env.children2parent[object_id]
                    parent = self.env.world.get_object(parent_ids[0])
                    if (
                        "openable" in parent.props
                        and parent.props["openable"]
                        and not parent.props["isOpen"]
                    ):
                        self.actions.append("Open" + "#" + parent.id)

            self.actions.append(plan["action"] + "#" + object_id)
            # self.execute(action=plan["action"], object_id=object_id)
            action = self.actions.pop(0)
        else:
            action = self.actions.pop(0)
        return action

    def get_observation_diff(self, starts, targets):
        diff = []
        for start_obj, target_obj in zip(starts, targets):
            assert start_obj["name"] == target_obj["name"]
            if start_obj["position"] != target_obj["position"]:
                action_start = deepcopy(start_obj)
                action_target = deepcopy(target_obj)
                action_start["action"] = "PickUp"
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

    def generate_plan(self, starts, targets):
        diff = self.get_observation_diff(starts, targets)
        start = {
            "name": "agent",
            "object_id": "agent",
            "position": {"x": self.position.x, "y": self.position.y},
            "action": "start",
        }
        nodes = []
        nodes.append(start)
        nodes.extend(diff)
        shortest_path = None
        shortest_distance = float("inf")

        def distance(node1, node2):
            p1 = node1["position"]
            p2 = node2["position"]
            distance = math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)
            return distance

        def is_valid(path, node):
            if node["action"] == "Drop":
                find_pick = False
                for n in path:
                    if n["name"] == node["name"] and n["action"] == "PickUp":
                        find_pick = True
                        break
                if find_pick and path[-1]["action"] != node["action"]:
                    return True
            elif node["action"] == "PickUp":
                if path[-1]["action"] != node["action"]:
                    return True
            elif node["action"] == "Open":
                return True
            elif node["action"] == "start":
                return True
            return False

        def dfs(path, remaining_nodes, current_distance):
            nonlocal shortest_path, shortest_distance
            if not remaining_nodes:
                if current_distance < shortest_distance:
                    shortest_path = path
                    shortest_distance = current_distance
                return

            for node in remaining_nodes:
                if is_valid(path, node):
                    new_path = path + [node]
                    new_remaining_nodes = remaining_nodes.copy()
                    new_remaining_nodes.remove(node)
                    new_distance = (
                        current_distance + distance(path[-1], node) if path else 0
                    )
                    dfs(new_path, new_remaining_nodes, new_distance)

        for start_node in nodes:
            if start_node["action"] == "start":
                remaining_nodes = nodes.copy()
                remaining_nodes.remove(start_node)
                dfs([start_node], remaining_nodes, 0)
        shortest_path.pop(0)
        return shortest_path

    def step(self, action_dict):
        parsed_response = {}
        expert_action = self.query_expert()
        print(expert_action)
        parsed_response = self.parse_expert_action(expert_action)
        logger.info(parsed_response)
        success = True
        if "action" in parsed_response and parsed_response["action"] != "UserInput":
            if parsed_response["action"] == "Pass":
                parsed_response["feedback"] = self.env.feedback_builder.build("Pass")
                success = False
            elif parsed_response["action"] == "Stop":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "Stop", "default"
                )
                success = True
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

    def parse_expert_action(self, expert_action):
        """
        ['Start', TurnRight', 'MoveAhead', 'TurnLeft', "Pickup|obj_id", "Drop|obj_id", "Open|obj_id"]
        """
        obj_id = None
        obj_name = None
        action_trans = ""
        action = ""
        if not self.status["started"]:
            self.status["started"] = True
            return dict(
                success=True,
                response=[("act", "YES.")],
                feedback=self.env.feedback_builder.build(
                    "Start",
                    original_state=self.env.target_pose_description,
                    observation=self.env.get_agent_position_observation(self)
                    + self.env.get_observation(self),
                ),
            )
        else:
            if "MoveAhead" in expert_action:
                action = "MoveAhead"
                action_trans = "move_ahead"
            elif "TurnLeft" in expert_action:
                action = "TurnLeft"
                action_trans = "turn_left"
            elif "TurnRight" in expert_action:
                action = "TurnRight"
                action_trans = "turn_right"
            elif "PickUp" in expert_action:
                obj_id = expert_action.split("#")[1]
                obj_name = self.env.object_id2name[obj_id]
                action = "PickUp"
                action_trans = "pick_up" + " [" + obj_name + "]"
            elif "Drop" in expert_action:
                obj_id = expert_action.split("#")[1]
                obj_name = self.env.object_id2name[obj_id]
                action = "Drop"
                action_trans = "drop" + " [" + obj_name + "]"
            elif "Open" in expert_action:
                obj_id = expert_action.split("#")[1]
                obj_name = self.env.object_id2name[obj_id]
                action = "Open"
                action_trans = "open" + " [" + obj_name + "]"
            elif "stop" in expert_action and len(expert_action) < 20:
                action = "Stop"
                action_trans = "stop []"
            else:
                action = "Pass"
                action_trans = "pass"
            if obj_id:
                action_arg = {"object_id": obj_id}
            else:
                action_arg = {}
            response = []
            response.append(("act", action_trans))
            return dict(response=response, action=action, action_arg=action_arg)
