# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import math

from langsuite.actions.base_action import ACTION_REGISTERY, ActionFeedback, BaseAction
from langsuite.shapes import Line2D, Point2D


@ACTION_REGISTERY.register()
class MoveBackAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        distance = kwargs.get("distance", 1) * self.agent.step_size
        dest = self.agent.position - self.agent.view_vector * distance

        traj = Line2D([self.agent.position, dest])

        if not self.env.is_valid_trajectory(traj):
            observation = self.env.get_observation(self.agent)
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "MoveBack", "failure.isBlocked", observation=observation
                ),
            )
        else:
            self.agent.set_position(dest)
            observation = self.env.get_observation(self.agent)
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "MoveBack",
                    "success.default",
                    distance=distance,
                    observation=observation,
                ),
            )


@ACTION_REGISTERY.register()
class PanLeftAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["degree"]

    def exec(self, **kwargs):
        # degree = kwargs.get("degree", self.degree)
        degree = 90
        # TODO
        self.agent.view_vector.rotate(-degree)
        distance = kwargs.get("distance", 1) * self.agent.step_size
        dest = self.agent.position + self.agent.view_vector * distance
        self.agent.view_vector.rotate(degree)

        traj = Line2D([self.agent.position, dest])
        observation = self.env.get_observation(self.agent)
        if not self.env.is_valid_trajectory(traj):
            self.env.is_valid_trajectory(traj)
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "PanLeft", "failure.isBlocked", observation=observation
                ),
            )
        else:
            self.agent.set_position(dest)
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "PanLeft",
                    "success.default",
                    distance=distance,
                    observation=observation,
                ),
            )


@ACTION_REGISTERY.register()
class PanRightAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent)

    def exec(self, **kwargs):
        # degree = kwargs.get("degree", self.degree)
        degree = 90
        # TODO
        self.agent.view_vector.rotate(degree)
        distance = kwargs.get("distance", 1) * self.agent.step_size
        dest = self.agent.position + self.agent.view_vector * distance
        self.agent.view_vector.rotate(-degree)

        traj = Line2D([self.agent.position, dest])
        observation = self.env.get_observation(self.agent)
        if not self.env.is_valid_trajectory(traj):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "PanRight", "failure.isBlocked", observation=observation
                ),
            )
        else:
            self.agent.set_position(dest)
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "PanRight",
                    "success.default",
                    distance=distance,
                    observation=observation,
                ),
            )


@ACTION_REGISTERY.register()
class PlaceAction(BaseAction):
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

        if "object_id" in kwargs:
            object_id = kwargs.get("object_id")

        if "object" in kwargs:
            object_id = kwargs.get("object").id

        if len(self.agent.inventory) != 1:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Place",
                    "failure.inventoryNotOne",
                    observation=self.env.get_observation(self.agent),
                ),
            )
        obj = self.agent.inventory[0]
        obj_name = self.env.object_id2name[obj.id]
        receptacle_name = self.env.object_id2name[object_id]

        if not self.env.world.contains_object(object_id):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.invalidObjectName",
                    object=obj_name,
                    receptacle_name=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        receptacle = self.env.world.get_object(object_id)

        if not self.agent.can_observe(receptacle.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    object=obj_name,
                    receptacle_name=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        # if not self.agent.can_manipulate(receptacle.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "Place",
        #             "failure.notInManipulate",
        #             receptacle_name=receptacle_name,
        #             observation=self.env.get_observation(self.agent),
        #         ),
        #     )

        self.agent.inventory.remove(obj)
        obj.props["parentReceptacles"].append(receptacle.id)
        if receptacle.props["receptacleObjectIds"] is None:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Place",
                    "failure.receptacleNotPlaceable",
                    object=obj_name,
                    receptacle_name=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        receptacle.props["receptacleObjectIds"].append(obj.id)
        receptacle.children[obj.id] = obj
        obj.update_position(receptacle.position)

        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "Place",
                "success.default",
                object=obj_name,
                receptacle_name=receptacle_name,
                observation=self.env.get_observation(self.agent),
            ),
        )


@ACTION_REGISTERY.register()
class PourAction(BaseAction):
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
        # manipulation_distance = kwargs.get(
        #     "manipulation_distance", self.manipulation_distance
        # )

        if "object_id" not in kwargs and "object" not in kwargs:
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

        if len(self.agent.inventory) != 1:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Pour",
                    "failure.inventoryNotOne",
                    observation=self.env.get_observation(self.agent),
                ),
            )
        obj = self.agent.inventory[0]
        obj_name = self.env.object_id2name[obj.id]
        receptacle_name = self.env.object_id2name[object_id]

        if not self.env.world.contains_object(object_id):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.invalidObjectName",
                    object=obj_name,
                    receptacle_name=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        receptacle = self.env.world.get_object(object_id)
        if not self.agent.can_observe(receptacle.geometry):
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotInView",
                    object=obj_name,
                    receptacle_name=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        if obj.props["isFilledWithLiquid"] is False:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "Pour",
                    "failure.notFilledWithLiquid",
                    object=obj_name,
                    receptacle_name=receptacle_name,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        obj.props["isFilledWithLiquid"] = False
        if receptacle.props["canFillWithLiquid"] is True:
            receptacle.props["isFilledWithLiquid"] = True

        # if not self.agent.can_manipulate(receptacle.position, manipulation_distance):
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "Place",
        #             "failure.notInManipulate",
        #             receptacle_name=receptacle_name,
        #             observation=self.env.get_observation(self.agent),
        #         ),
        #     )

        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "Pour",
                "success.default",
                object=obj_name,
                receptacle_name=receptacle_name,
                observation=self.env.get_observation(self.agent),
            ),
        )


@ACTION_REGISTERY.register()
class OpenProgressCheckAction(BaseAction):
    def __init__(self, env, agent, manipulation_distance=None) -> None:
        super().__init__(
            env=env, agent=agent, manipulation_distance=manipulation_distance
        )
        self.kwlist = ["object_id", "manipulation_distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        original_format_objects = []
        for o in self.env.world.id2object.values():
            original_format_objects.append(o.props)
        progress_check_output = self.env.task_to_check.check_episode_progress(
            original_format_objects
        )
        task_desc, success, subgoals, gc_total, gc_satisfied = [
            progress_check_output["description"],
            progress_check_output["success"],
            progress_check_output["subgoals"],
            progress_check_output["goal_conditions_total"],
            progress_check_output["goal_conditions_satisfied"],
        ]
        condition_failure_list = []
        for subgoal in subgoals:
            subgoal["success"] = 1 if subgoal["success"] else 0
            if "step_successes" in subgoal:
                subgoal["step_successes"] = [int(v) for v in subgoal["step_successes"]]
            for step in subgoal["steps"]:
                step["success"] = 1 if step["success"] else 0
                if "desc" in step:
                    condition_failure_list.append(step["desc"])
        condition_failure_descs = "\n".join(condition_failure_list)

        if success:
            return ActionFeedback(
                success=True,
                feedback=self.feedback_builder.build(
                    "OpenProgressCheck", "success.default"
                ),
            )
        else:
            return ActionFeedback(
                success=False,
                feedback=self.feedback_builder.build(
                    "OpenProgressCheck",
                    "failure.default",
                    goal_conditions_total=gc_total,
                    goal_conditions_satisfied=gc_satisfied,
                    condition_failure_descs=condition_failure_descs,
                ),
            )


@ACTION_REGISTERY.register()
class SelectOidAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent, degree=degree)
        self.kwlist = ["object_id"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        object_id = kwargs.get("object_id")
        obj = self.env.world.get_object(object_id)
        dx = obj.position.x - self.agent.position.x
        dy = obj.position.y - self.agent.position.y
        direction = Point2D(1 if dx > 0 else -1, 1 if dy > 0 else -1)
        agent_towards = self.agent.view_vector
        # use dx, dz to determine which direction to turn
        radian = math.atan2(agent_towards.y, agent_towards.x) - math.atan2(
            direction.y, direction.x
        )
        degree = (math.degrees(radian) + 360) % 360
        location = ""
        if degree == 135:
            location = "at the left rear of follower"
        elif degree == 225:
            location = "at the right rear of follower"
        elif degree == 315:
            location = "in front and right of follower"
        elif degree == 45:
            location = "in front and left of follower"

        return ActionFeedback(
            success=True,
            feedback=self.feedback_builder.build(
                "SelectOid",
                "success.default",
                object=self.env.object_id2name[object_id],
                location=location,
                observation=self.env.get_observation(self.agent),
            ),
        )
