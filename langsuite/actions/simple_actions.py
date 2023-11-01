# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from copy import deepcopy

from langsuite.actions.base_action import ACTION_REGISTERY, ActionFeedback, BaseAction
from langsuite.shapes import Line2D, Point2D
from langsuite.utils.grid_world import *
from langsuite.utils.logging import logger


@ACTION_REGISTERY.register()
class MoveAheadAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        distance = kwargs.get("distance", 1) * self.agent.step_size
        dest = self.agent.position + self.agent.view_vector * distance

        traj = Line2D([self.agent.position, dest])

        if not self.env.is_valid_trajectory(traj):
            # sys.env.render()
            self.env.is_valid_trajectory(traj)
            observation = self.env.get_observation(self.agent)
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "MoveAhead", "failure.isBlocked", observation=observation
                ),
            )
        else:
            self.agent.set_position(dest)
            observation = self.env.get_observation(self.agent)
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "MoveAhead",
                    "success.default",
                    distance=distance,
                    observation=observation,
                ),
            )


@ACTION_REGISTERY.register()
class GoToAction(BaseAction):
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
                    "GoTo",
                    "failure.objectNotFound",
                    object_id=object_id,
                    observation=self.env.get_observation(self.agent),
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
                    "GoTo",
                    "failure.alreadyAtTarget",
                    object=self.env.object_id2name[object_id],
                    observation=self.env.get_observation(self.agent),
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
                    "GoTo",
                    "failure.default",
                    object=self.env.object_id2name[object_id],
                    observation=self.env.get_observation(self.agent),
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

        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "GoTo",
                "success.default",
                object=self.env.object_id2name[object_id],
                observation=self.env.get_observation(self.agent),
                symbolic_action_list=action_trajectory,
            ),
        )


@ACTION_REGISTERY.register()
class TurnLeftAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent, degree=degree)
        self.kwlist = ["degree"]

    def exec(self, **kwargs):
        degree = kwargs.get("degree", self.degree)
        # TODO
        self.agent.rotate(-degree)
        observation = self.env.get_observation(self.agent)
        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "TurnLeft", "success.default", degree=degree, observation=observation
            ),
        )


@ACTION_REGISTERY.register()
class TurnRightAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent, degree=degree)

    def exec(self, **kwargs):
        degree = kwargs.get("degree", self.degree)
        # TODO
        self.agent.rotate(degree)
        observation = self.env.get_observation(self.agent)
        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "TurnRight", "success.default", degree=degree, observation=observation
            ),
        )


@ACTION_REGISTERY.register()
class PickUpAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        # self.env.render()
        logger.info("kwargs")
        logger.info(kwargs)
        manipulation_distance = kwargs.get(
            "manipulation_distance", self.agent.max_manipulate_distance
        )

        if kwargs.get("object_id") is None and len(kwargs.get("object_name", "")) > 0:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.invalidObjectName",
                    object=kwargs.get("object_name"),
                    observation=self.env.get_observation(self.agent),
                ),
            )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
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
                    observation=self.env.get_observation(self.agent),
                ),
            )

        obj_name = self.env.object_id2name[object_id]
        obj = self.env.world.get_object(object_id)
        observed_objects = self.env.get_observed_objects(self.agent)
        if obj.id not in observed_objects:
            observation = self.env.get_observation(self.agent)
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    max_view_steps=self.agent.max_view_distance / self.agent.step_size,
                    object=obj_name,
                    observation=observation,
                ),
            )
        # TODO
        # if not self.agent.can_observe(obj.geometry):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build("PickUp", "failure.notInView", object=object_id)
        #     )

        # if not self.agent.can_manipulate(obj.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "InvalidAction",
        #             "failure.objectNotInMainpulation",
        #             object=obj_name,
        #             manipulation_distance=manipulation_distance,
        #             observation=self.env.get_observation(self.agent),
        #         ),
        #     )
        if "pickupable" in obj.props and obj.props["pickupable"]:
            if (
                self.agent.add_inventory(obj)
                and self.env.world.pop_object(object_id) is not None
            ):
                return ActionFeedback(
                    success=True,
                    feedback=self.feedback_builder.build(
                        "PickUp",
                        "success.default",
                        object=obj_name,
                        observation=self.env.get_observation(self.agent),
                        inventory=self.agent.get_inventory(),
                    ),
                )
            else:
                return ActionFeedback(
                    success=False,
                    feedback=self.feedback_builder.build(
                        "PickUp",
                        "failure.intentoryFilled",
                        object=obj_name,
                        observation=self.env.get_held_object_observation(self.agent),
                    ),
                )
        else:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "PickUp",
                    "failure.notPickupable",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )


@ACTION_REGISTERY.register()
class PutAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        manipulation_distance = kwargs.get(
            "manipulation_distance", self.manipulation_distance
        )

        if "object_id" not in kwargs and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
                ),
            )

        if "receptacle_id" not in kwargs and "receptacle" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
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
        obj = self.agent.get_object_in_inventory(object_id)
        if not obj:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Put",
                    "failure.NotInInventory",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                    inventory=self.agent.get_inventory(),
                ),
            )

        if not self.env.world.contains_object(receptacle_id):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotExist",
                    object=receptacle_id,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        receptacle = self.env.world.get_object(receptacle_id)
        receptacle_name = self.env.object_id2name[receptacle_id]
        if not obj or not receptacle:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Put",
                    "failure.default",
                    object=obj_name,
                    receptacle=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        observed_objects = self.env.get_observed_objects(self.agent)
        if receptacle.id not in observed_objects:
            # if not self.agent.can_observe(receptacle.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Put",
                    "failure.notInView",
                    receptacle=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        # if not self.agent.can_manipulate(receptacle.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=f"Object {receptacle_id} is not in manipulation distance.",
        #     )

        self.agent.inventory.remove(obj)
        obj.position = receptacle.position
        receptacle.children[obj.id] = obj

        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "Put",
                "success.default",
                object=obj_name,
                receptacle=receptacle_name,
                observation=self.env.get_observation(self.agent),
            ),
        )


@ACTION_REGISTERY.register()
class DropAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        if len(self.agent.inventory) == 0:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Drop",
                    "failure.emptyInventory",
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            # TODO: drop object, not change the position, the drop position can be the grid in front of the agent.
            obj = self.agent.inventory.pop(0)
            drop_direc = deepcopy(self.agent.view_vector)
            drop_direc.rotate(45)
            dest = self.agent.position + drop_direc * 0.5
            obj.update(dest)
            self.env.world.add_object(obj)
            obj_name = self.env.object_id2name[obj.id]
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "Drop",
                    "success.default",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                    inventory=self.agent.get_inventory(),
                ),
            )


@ACTION_REGISTERY.register()
class OpenAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "openness", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        manipulation_distance = kwargs.get(
            "manipulation_distance", self.manipulation_distance
        )

        if kwargs.get("object_id") is None and len(kwargs.get("object_name", "")) > 0:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.invalidObjectName",
                    object=kwargs.get("object_name"),
                    observation=self.env.get_observation(self.agent),
                ),
            )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
                ),
            )

        openness = kwargs.get("openness", 1.0)

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
                    observation=self.env.get_observation(self.agent),
                ),
            )

        obj_name = self.env.object_id2name[object_id]
        obj = self.env.world.get_object(object_id)
        observed_objects = self.env.get_observed_objects(self.agent)
        if obj.id not in observed_objects:
            # if not self.agent.can_observe(obj.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    max_view_steps=self.agent.max_view_distance / self.agent.step_size,
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        if not self.agent.can_manipulate(obj.position, manipulation_distance):
            return ActionFeedback(
                success=False,
                feedback=f"Object {obj_name} is not in manipulation distance.",
            )

        if "openable" in obj.props and obj.props["openable"]:
            obj.props["openness"] = openness
            obj.props["isOpen"] = True
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "Open",
                    "success.default",
                    object=obj_name,
                    openness=openness,
                    observation=self.env.get_openned_object_observation(object_id),
                ),
            )
        return ActionFeedback(
            success=False,
            feedback=self.feedback_builder.build(
                "Open",
                "failure.notOpenable",
                object=obj_name,
                observation=self.env.get_observation(self.agent),
            ),
        )


@ACTION_REGISTERY.register()
class ToggleOnAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        manipulation_distance = kwargs.get(
            "manipulation_distance", self.manipulation_distance
        )

        if kwargs.get("object_id") is None and len(kwargs.get("object_name", "")) > 0:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.invalidObjectName",
                    object=kwargs.get("object_name"),
                    observation=self.env.get_observation(self.agent),
                ),
            )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
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
                    object=obj_name or object_id,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        obj_name = self.env.object_id2name[object_id]
        obj = self.env.world.get_object(object_id)
        observed_objects = self.env.get_observed_objects(self.agent)
        if obj.id not in observed_objects:
            # if not self.agent.can_observe(obj.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    max_view_steps=self.agent.max_view_distance / self.agent.step_size,
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        # if not self.agent.can_manipulate(obj.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "InvalidAction",
        #             "failure.objectNotInMainpulation",
        #             object=obj_name,
        #             observation=self.env.get_observation(self.agent),
        #         ),
        #     )

        if "toggleable" in obj.props and obj.props["toggleable"]:
            obj.props["isToggled"] = True
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "ToggleOn",
                    "success.default",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "ToggleOn",
                    "failure.notToggleable",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )


@ACTION_REGISTERY.register()
class CloseAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        manipulation_distance = kwargs.get(
            "manipulation_distance", self.manipulation_distance
        )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
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
                    "InvalidAction", "failure.objectNotExist", object=object_id
                ),
            )

        obj_name = self.env.object_id2name[object_id]
        obj = self.env.world.get_object(object_id)
        observed_objects = self.env.get_observed_objects(self.agent)
        if obj.id not in observed_objects:
            # if not self.agent.can_observe(obj.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    max_view_steps=self.agent.max_view_distance / self.agent.step_size,
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        # if not self.agent.can_observe(obj.geometry):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build("Close", "failure.notInView", object=object_id)
        #     )

        # if not self.agent.can_manipulate(obj.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "InvalidAction",
        #             "failure.objectNotInMainpulation",
        #             object=obj_name,
        #             observation=self.env.get_observation(self.agent),
        #         ),
        #     )

        if "openable" in obj.props and obj.props["openable"]:
            obj.props["openness"] = 0.0
            obj.props["isOpen"] = False
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "Close",
                    "success.default",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Close",
                    "failure.notCloseable",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )


@ACTION_REGISTERY.register()
class ToggleOffAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        manipulation_distance = kwargs.get(
            "manipulation_distance", self.manipulation_distance
        )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
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
                    "InvalidAction", "failure.objectNotExist", object=object_id
                ),
            )

        obj_name = self.env.object_id2name[object_id]
        obj = self.env.world.get_object(object_id)
        observed_objects = self.env.get_observed_objects(self.agent)
        if obj.id not in observed_objects:
            # if not self.agent.can_observe(obj.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    max_view_steps=self.agent.max_view_distance / self.agent.step_size,
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        # if not self.agent.can_observe(obj.geometry):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build("Close", "failure.notInView", object=object_id)
        #     )

        # if not self.agent.can_manipulate(obj.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "InvalidAction",
        #             "failure.objectNotInMainpulation",
        #             object=obj_name,
        #             observation=self.env.get_observation(self.agent),
        #         ),
        #     )

        if "toggleable" in obj.props and obj.props["toggleable"]:
            obj.props["isToggled"] = False
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "ToggleOff",
                    "success.default",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "ToggleOff",
                    "failure.notToggleable",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )


@ACTION_REGISTERY.register()
class SliceAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_id: name of object
        """
        manipulation_distance = kwargs.get(
            "manipulation_distance", self.manipulation_distance
        )

        if kwargs.get("object_id") is None and "object" not in kwargs:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
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
                    "InvalidAction", "failure.objectNotExist", object=object_id
                ),
            )
        obj_name = self.env.object_id2name[object_id]
        obj = self.env.world.get_object(object_id)
        observed_objects = self.env.get_observed_objects(self.agent)
        if obj.id not in observed_objects:
            # if not self.agent.can_observe(obj.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    max_view_steps=self.agent.max_view_distance / self.agent.step_size,
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        # if not self.agent.can_manipulate(obj.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "InvalidAction",
        #             "failure.objectNotInMainpulation",
        #             object=obj_name,
        #             observation=self.env.get_observation(self.agent),
        #         ),
        #     )

        if "sliceable" in obj.props and obj.props["sliceable"]:
            if obj.props["isSliced"]:
                return ActionFeedback(
                    success=False,
                    feedback=self.feedback_builder.build(
                        "Slice",
                        "failure.objectIsSliced",
                        object=obj_name,
                        observation=self.env.get_observation(self.agent),
                    ),
                )
            else:
                obj.props["isSliced"] = True
                for i in range(10):
                    # TODO change position slightly
                    category = obj.id.split("|")[0]
                    sliced_obj = deepcopy(obj)
                    sliced_obj.id = obj.id + "|" + category + "Sliced_" + str(i)
                    sliced_obj.props["sliceable"] = False
                    sliced_obj.props["isSliced"] = True
                    sliced_obj.props["cookable"] = True
                    sliced_obj.props["objectType"] = (
                        sliced_obj.props["objectType"] + "Sliced"
                    )
                    self.env.world.add_object(sliced_obj)
                    self.env.world.id2object[sliced_obj.id] = sliced_obj

                    obj_category_dic = self.env.obj_category_dic
                    obj_index = len(obj_category_dic[category])
                    name = category + "|Sliced" + "_" + str(i)
                    name = name.lower()
                    self.env.object_id2name[sliced_obj.id] = name
                    self.env.object_name2id[name] = sliced_obj.id
                return ActionFeedback(
                    success=True,
                    feedback=self.feedback_builder.build(
                        "Slice",
                        "success.default",
                        object=obj_name,
                        observation=self.env.get_observation(self.agent),
                    ),
                )
        else:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Slice",
                    "failure.notSliceable",
                    object=obj_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )


@ACTION_REGISTERY.register()
class ChatAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        # if "chat_response" in kwargs:
        #     chat_response = kwargs.get("chat_response")
        # else:
        #     chat_prompt = self.feedback_builder.build(
        #         "intro",
        #         "message_instruction_for_" + self.agent.agent_name,
        #     )
        #     prompt = (
        #         self.agent.current_prompt + chat_prompt + f"{self.agent.agent_name}: "
        #     )
        #     chat_response = self.agent.fetch_prompt_response(prompt)

        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "Chat",
                "success.default",
                # agent_name=self.agent.agent_name,
                # chat_response=chat_response,
            ),
        )


@ACTION_REGISTERY.register()
class NoOpAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        return ActionFeedback(success=True, feedback="Do nothing, wait")
