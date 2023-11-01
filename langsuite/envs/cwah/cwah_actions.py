# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from langsuite.actions import ActionFeedback, get_action
from langsuite.actions.base_action import ACTION_REGISTERY, ActionFeedback, BaseAction
from langsuite.shapes import Point2D
from langsuite.utils.grid_world import *
from langsuite.utils.logging import logger


@ACTION_REGISTERY.register()
class CwahGoToAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent, degree=degree)
        self.kwlist = ["object_id"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        object_id = kwargs.get("object_id")
        if isinstance(self.env.world.room_polygons, dict):
            x_min, x_max, y_min, y_max = (
                float("inf"),
                -float("inf"),
                float("inf"),
                -float("inf"),
            )
            for room_polygons in self.env.world.room_polygons.values():
                (
                    x_min_one_room,
                    x_max_one_room,
                    y_min_one_room,
                    y_max_one_room,
                ) = cal_wall_min_max(room_polygons)
                x_min = min(x_min, x_min_one_room)
                x_max = max(x_max, x_max_one_room)
                y_min = min(y_min, y_min_one_room)
                y_max = max(y_max, y_max_one_room)
        else:
            x_min, x_max, y_min, y_max = cal_wall_min_max(self.env.world.room_polygons)
        grid_world = GridWorld(x_min, x_max, y_min, y_max, self.agent.step_size)
        agent_direction = get_direction(self.agent.view_vector)
        for obj_id, obj in self.env.world.objects.items():
            if "Floor" in obj.category:
                x_list = list(obj.geometry.shapely_geo.exterior.xy[0][:4])
                y_list = list(obj.geometry.shapely_geo.exterior.xy[1][:4])

                grid_world.add_object(obj.position.x, obj.position.y, (x_list, y_list))
        object = self.env.world.id2object.get(object_id)
        if object is None:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "CwahGoTo",
                    "failure.objectNotFound",
                    object=object,
                    observation=self.agent.get_progress(),
                ),
            )
        end_coordinate, action_trajectory, opetation_rotate_list = grid_world.get_path(
            self.agent.position,
            (object.position.x, object.position.y),
            agent_direction,
            grid_world.grid,
        )
        if end_coordinate is False:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "CwahGoTo",
                    "failure.alreadyAtTarget",
                    object=self.env.object_id2name[object_id],
                    observation=self.agent.get_progress(),
                    symbolic_action_list=action_trajectory,
                ),
            )
        if (
            end_coordinate is None
            and action_trajectory is None
            and opetation_rotate_list is None
        ):
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "CwahGoTo",
                    "failure.default",
                    object=self.env.object_id2name[object_id],
                    observation=self.agent.get_progress(),
                    symbolic_action_list=action_trajectory,
                ),
            )
        dest = Point2D(end_coordinate[0], end_coordinate[1])
        self.agent.set_position(dest)
        for rotate_action in opetation_rotate_list:
            if rotate_action == "TurnRight":
                self.agent.rotate(self.degree)
            elif rotate_action == "TurnLeft":
                self.agent.rotate(-self.degree)
        self.agent._update()
        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "CwahGoTo",
                "success.default",
                object=self.env.object_id2name[object_id],
                observation=self.agent.get_progress(),
                symbolic_action_list=action_trajectory,
            ),
        )


@ACTION_REGISTERY.register()
class CwahPickUpAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["object_id"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        logger.info("kwargs")
        logger.info(kwargs)

        if kwargs.get("object_id") is None and len(kwargs.get("object_name", "")) > 0:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.invalidObjectName",
                    object=kwargs.get("object_name"),
                    observation=self.agent.get_progress(),
                ),
            )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.agent.get_progress(),
                ),
            )

        if "object_id" in kwargs:
            object_id = kwargs.get("object_id")

        if "object" in kwargs:
            object_id = kwargs.get("object").id

        if not self.env.world.contains_object(object_id):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotExist",
                    object=object_id,
                    observation=self.agent.get_progress(),
                ),
            )

        obj_name = self.env.object_id2name[object_id]
        if "cupcake" in obj_name:
            print(self.agent.reachable_objects)
        if obj_name not in self.agent.reachable_objects:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.notClose",
                    object=obj_name,
                    observation=self.agent.get_progress(),
                ),
            )

        obj_node = self.agent.id2node[object_id]
        if "GRABBABLE" not in obj_node["properties"]:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "CwahPickUp",
                    "failure.notGrabbable",
                    object=obj_name,
                    observation=self.agent.get_progress(),
                ),
            )

        if len(self.agent.grabbed_objects) < 2:
            possible_hands = ["HOLDS_RH", "HOLDS_LH"]
            for e in self.env.full_graph["edges"]:
                x, r, y = e["from_id"], e["relation_type"], e["to_id"]
                if x != self.agent.agent_id:
                    continue
                if r in ["HOLDS_LH", "HOLDS_RH"]:
                    possible_hands.remove(r)
                    break

            for e in self.env.full_graph["edges"]:
                x, r, y = e["from_id"], e["relation_type"], e["to_id"]
                if x == self.agent.agent_id and y == object_id:
                    if r == "CLOSE":
                        e["relation_type"] = possible_hands[0]
                        break

            self.agent._update()

            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "CwahPickUp",
                    "success.default",
                    object=obj_name,
                    observation=self.agent.get_progress(),
                ),
            )
        else:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "CwahPickUp",
                    "failure.intentoryFilled",
                    object=obj_name,
                    observation=self.agent.get_progress(),
                ),
            )


@ACTION_REGISTERY.register()
class CwahPutAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["object_id"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """

        if "object_id" not in kwargs and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "CwahPut",
                    "failure.objectNotProvide",
                    observation=self.agent.get_progress(),
                ),
            )

        if "receptacle_id" not in kwargs and "receptacle" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.agent.get_progress(),
                ),
            )

        if "object_id" in kwargs:
            object_id = kwargs.get("object_id")

        if "object" in kwargs:
            object_id = kwargs.get("object").id

        if "receptacle_id" in kwargs:
            receptacle_id = kwargs.get("receptacle_id")

        if "receptacle" in kwargs:
            receptacle_id = kwargs.get("receptacle").id

        obj_name = self.env.object_id2name[object_id]
        receptacle = self.env.world.get_object(receptacle_id)
        receptacle_name = self.env.object_id2name[receptacle_id]

        obj_node = self.agent.id2node[object_id]
        if object_id not in self.agent.grabbed_objects:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "CwahPut",
                    "failure.NotInInventory",
                    object=obj_name,
                    receptacle=receptacle_name,
                    observation=self.agent.get_progress(),
                ),
            )
        receptacle_node = self.agent.id2node[receptacle.id]
        relation_type = "NONE"
        if "CONTAINERS" in receptacle_node["properties"]:
            existed = False
            for e in self.env.full_graph["edges"]:
                x, r, y = e["from_id"], e["relation_type"], e["to_id"]
                if x == object_id and y == receptacle_id:
                    e["relation_type"] = "INSIDE"
                    existed = True
            if not existed:
                self.env.full_graph["edges"].append(
                    {
                        "from_id": object_id,
                        "relation_type": "INSIDE",
                        "to_id": receptacle_id,
                    }
                )
            relation_type = "INSIDE"
        else:
            existed = False
            for e in self.env.full_graph["edges"]:
                x, r, y = e["from_id"], e["relation_type"], e["to_id"]
                if x == object_id and y == receptacle_id:
                    e["relation_type"] = "ON"

                    existed = True
            if not existed:
                self.env.full_graph["edges"].append(
                    {
                        "from_id": object_id,
                        "relation_type": "ON",
                        "to_id": receptacle_id,
                    }
                )
            relation_type = "ON"

        for e in self.env.full_graph["edges"]:
            x, r, y = e["from_id"], e["relation_type"], e["to_id"]
            if x != self.agent.agent_id or y != object_id:
                continue
            if r in ["HOLDS_RH", "HOLDS_LH"]:
                e["relation_type"] = "CLOSE"
                break
        self.agent._update()
        obj_node["obj_transform"]["position"] = receptacle_node["obj_transform"][
            "position"
        ]

        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "CwahPut",
                "success.default",
                object=self.env.object_id2name[object_id],
                receptacle=self.env.object_id2name[receptacle_id],
                observation=self.agent.get_progress(),
                relation_type=relation_type.lower(),
            ),
        )


@ACTION_REGISTERY.register()
class CwahOpenAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["object_id"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """

        if kwargs.get("object_id") is None and len(kwargs.get("object_name", "")) > 0:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.invalidObjectName",
                    object=kwargs.get("object_name"),
                    observation=self.agent.get_progress(),
                ),
            )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.agent.get_progress(),
                ),
            )

        if "object_id" in kwargs:
            object_id = kwargs.get("object_id")

        if "object" in kwargs:
            object_id = kwargs.get("object").id
        obj_name = self.env.object_id2name[object_id]
        if not self.env.world.contains_object(object_id):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotExist",
                    observation=self.agent.get_progress(),
                ),
            )

        obj_name = self.env.object_id2name[object_id]
        obj_node = self.agent.id2node[object_id]
        if "CAN_OPEN" not in obj_node["properties"]:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "CwahOpen",
                    "failure.notOpenable",
                    object=obj_name,
                    observation=self.agent.get_progress(),
                ),
            )
        if obj_name not in self.agent.reachable_objects:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.notClose",
                    max_view_steps=self.agent.max_view_distance / self.agent.step_size,
                    object=obj_name,
                    observation=self.agent.get_progress(),
                ),
            )

        obj_node = self.agent.id2node[object_id]
        if "CAN_OPEN" in obj_node["properties"] and "CLOSED" in obj_node["states"]:
            obj_node["states"].remove("CLOSED")
            obj_node["states"].append("OPEN")
            self.agent._update()
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "CwahOpen",
                    "success.default",
                    object=obj_name,
                    observation=self.env.get_openned_object_observation(object_id),
                ),
            )

        return ActionFeedback(
            success=False,
            feedback=self.feedback_builder.build(
                "CwahOpen",
                "failure.notOpenable",
                object=obj_name,
                observation=self.agent.get_progress(),
            ),
        )


@ACTION_REGISTERY.register()
class GoExploreAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)

    def exec(self, **kwargs):
        id = kwargs["object_id"]
        class_name = self.agent.id2node[id]["class_name"]
        observation = self.agent.get_progress()
        if id == self.agent.current_room["id"]:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "GoExplore",
                    "failure.alreadyAtTargetRoom",
                    room=self.env.object_id2name[id],
                    observation=observation,
                ),
            )
        elif class_name not in self.agent.rooms_name:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "GoExplore",
                    "failure.notRoom",
                    # room=self.env.object_id2name[id],
                    observation=observation,
                ),
            )
        go_to = get_action(action_name="CwahGoTo", env=self.env, agent=self.agent)
        return go_to.step(**kwargs)


@ACTION_REGISTERY.register()
class GoCheckAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)

    def exec(self, **kwargs):
        # assert len(self.agent.grabbed_objects) < 2  # must have at least one free hands
        go_to = get_action(action_name="CwahGoTo", env=self.env, agent=self.agent)
        open = get_action(action_name="CwahOpen", env=self.env, agent=self.agent)
        observation = self.agent.get_progress()
        target_container_id = kwargs["object_id"]
        target_container_name = self.env.object_id2name[target_container_id]
        # if self.agent.id2node[target_container_id]["class_name"] in self.env.all_room_name:
        #     target_container_room = self.agent.id2node[target_container_id]
        # else:
        #     if target_container_id not in self.agent.id_inside_room:
        #         return ActionFeedback(
        #             success=False,
        #             feedback=self.feedback_builder.build(
        #                 "InvalidAction",
        #                 "failure.notFoundYet",
        #                 object=target_container_name,
        #                 observation=observation,
        #             ),
        #         )
        #     target_container_room = self.agent.id_inside_room[target_container_id]
        # if self.agent.current_room["class_name"] != target_container_room:
        #     return go_to.step(**kwargs)
        if target_container_id not in self.agent.id_inside_room:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.notFoundYet",
                    object=target_container_name,
                    observation=observation,
                ),
            )

        target_container = self.agent.id2node[target_container_id]
        if "OPEN" in target_container["states"]:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "GoCheck",
                    "failure.alreadyOpen",
                    object=target_container_name,
                    observation=observation,
                ),
            )
        if len(self.agent.grabbed_objects) >= self.agent.inventory_capacity:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "GoCheck",
                    "failure.noFreehand",
                    observation=observation,
                ),
            )
        if target_container_name in self.agent.reachable_objects:
            return open.step(**kwargs)
        else:
            go_to.step(**kwargs)
            return open.step(**kwargs)


@ACTION_REGISTERY.register()
class GoGrabAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)

    def exec(self, **kwargs):
        go_to = get_action(action_name="CwahGoTo", env=self.env, agent=self.agent)
        pick_up = get_action(action_name="CwahPickUp", env=self.env, agent=self.agent)
        target_object_id = kwargs["object_id"]
        target_object_name = self.env.object_id2name[target_object_id]
        observation = self.agent.get_progress()
        oppo_agent = self.env.agents[1 - self.agent.agent_id]
        if (
            target_object_id in self.agent.grabbed_objects
            or target_object_id in oppo_agent.grabbed_objects
        ):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "GoGrab",
                    "failure.alreadyGrabbed",
                    object=target_object_name,
                    observation=observation,
                ),
            )
        # assert len(self.agent.grabbed_objects) < 2  # must have at least one free hands
        if target_object_id not in self.agent.id_inside_room:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.notFoundYet",
                    object=target_object_name,
                    observation=observation,
                ),
            )
        # target_object_room = self.agent.id_inside_room[target_object_id]
        # if self.agent.current_room["class_name"] != target_object_room:
        #     kwargs["object_id"] = self.agent.roomname2id[target_object_room]
        #     return go_to.step(**kwargs)

        # TODO: not here condition
        if target_object_name in self.agent.reachable_objects:
            return pick_up.step(**kwargs)
        else:
            go_to.step(**kwargs)
            return pick_up.step(**kwargs)


@ACTION_REGISTERY.register()
class GoPutAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)

    def exec(self, **kwargs):
        observation = self.agent.get_progress()
        go_to = get_action(action_name="CwahGoTo", env=self.env, agent=self.agent)
        put = get_action(action_name="CwahPut", env=self.env, agent=self.agent)
        if len(self.agent.grabbed_objects) == 0:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "GoPut",
                    "failure.emptyInventory",
                    observation=observation,
                ),
            )
        if type(self.agent.id_inside_room[self.agent.goal["goal_location_id"]]) is list:
            if len(self.agent.id_inside_room[self.agent.goal["goal_location_id"]]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.agent.id_inside_room[
                    self.agent.goal["goal_location_id"]
                ] = self.rooms_name[:]
            target_room_name = self.agent.id_inside_room[
                self.agent.goal["goal_location_id"]
            ][0]
        else:
            target_room_name = self.agent.id_inside_room[
                self.agent.goal["goal_location_id"]
            ]
        goal_name = self.env.object_id2name[self.agent.goal["goal_location_id"]]
        go_to_kwargs = {}
        # if self.agent.current_room["class_name"] != target_room_name:
        #     goto_kwargs["object_id"] = self.agent.roomname2id[target_room_name]
        #     return go_to.step(**kwargs)
        # if self.agent.goal["goal_location"] not in self.agent.reachable_objects:
        #     goto_kwargs["object_id"] = self.agent.goal["goal_location"].split(" ")[-1][1:-1]
        #     return go_to.step(**kwargs)
        if self.agent.goal["goal_location_id"] not in self.agent.id_inside_room:
            go_to_kwargs["object_id"] = self.agent.roomname2id[target_room_name]
            go_to.step(**go_to_kwargs)
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "GoPut",
                    "failure.searchForGoal",
                    object=goal_name,
                    room=target_room_name,
                    observation=observation,
                ),
            )
        object_id = self.agent.grabbed_objects[0]
        object_node = self.agent.id2node[object_id]
        receptacle_id = int(self.agent.goal["goal_location"].split(" ")[-1][1:-1])
        receptacle_node = self.agent.id2node[receptacle_id]

        if "CONTAINERS" in receptacle_node["properties"]:
            if (
                len(self.agent.grabbed_objects) < 2
                and "CLOSED" in receptacle_node["states"]
            ):
                open = get_action(
                    action_name="CwahOpen", env=self.env, agent=self.agent
                )
                open_kwargs = {}
                open_kwargs["object_id"] = receptacle_id
                open.step(**open_kwargs)

        put_kwargs = {}
        put_kwargs["object_id"] = object_id
        put_kwargs["receptacle_id"] = receptacle_id
        return put.step(**put_kwargs)
