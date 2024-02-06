# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from minigrid.core.world_object import Key

from langsuite.actions.base_action import (
    ACTION_REGISTERY,
    BabyAIActionFeedback,
    BaseAction,
)


@ACTION_REGISTERY.register()
class BabyAIMoveAheadAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["distance"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        fwd_pos = self.env.world.front_pos
        fwd_cell = self.env.world.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            _, _, done, _, _ = self.env.world.step(self.env.world.actions.forward)
            # self.agent.set_position(fwd_pos)
            # self.env.world.agent_pos = fwd_pos
            return BabyAIActionFeedback(
                success=True,
                task_success=done,
                feedback=self.feedback_builder.build(
                    "BabyAIMoveAhead",
                    "success.default",
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIMoveAhead",
                    "failure.isBlocked",
                    observation=self.env.get_observation(self.agent),
                ),
            )


@ACTION_REGISTERY.register()
class BabyAITurnLeftAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent, degree=degree)
        self.kwlist = ["degree"]

    def exec(self, **kwargs):
        # self.env.world.agent_dir -= 1
        # if self.env.world.agent_dir < 0:
        #     self.env.world.agent_dir += 4
        _, _, done, _, _ = self.env.world.step(self.env.world.actions.left)
        return BabyAIActionFeedback(
            success=True,
            task_success=done,
            feedback=self.feedback_builder.build(
                "BabyAITurnLeft",
                "success.default",
                observation=self.env.get_observation(self.agent),
            ),
        )


@ACTION_REGISTERY.register()
class BabyAITurnRightAction(BaseAction):
    def __init__(self, env, agent, degree=90) -> None:
        super().__init__(env=env, agent=agent, degree=degree)

    def exec(self, **kwargs):
        _, _, done, _, _ = self.env.world.step(self.env.world.actions.right)
        # self.env.world.agent_dir = (self.env.world.agent_dir + 1) % 4
        return BabyAIActionFeedback(
            success=True,
            task_success=done,
            feedback=self.feedback_builder.build(
                "BabyAITurnRight",
                "success.default",
                observation=self.env.get_observation(self.agent),
            ),
        )


@ACTION_REGISTERY.register()
class BabyAIPickUpAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
        type: type of object
        """
        if "object_type" in kwargs:
            object_type = kwargs.get("object_type")
        else:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
                ),
            )

        input_obj_des = object_type
        if "color" in kwargs:
            color = kwargs.get("color")
            input_obj_des = color + " " + object_type
        else:
            color = None
        target_pos = self.env.world.front_pos
        target_cell = self.env.world.grid.get(*target_pos)
        target_agent_pos = self.env.world.agent_pos
        # step_count = 0
        # target_cell = None
        # for i in range(1, self.agent.view_size):
        #     next_pos = self.env.world.agent_pos + self.env.world.dir_vec * i
        #     if (
        #         next_pos[0] < self.env.world.width
        #         and next_pos[1] < self.env.world.height
        #     ):
        #         next_cell = self.env.world.grid.get(*next_pos)

        #         if next_cell is None or next_cell.can_overlap():
        #             step_count += 1
        #             target_agent_pos = next_pos
        #             continue
        #         else:
        #             target_cell = next_cell
        #             target_pos = next_pos
        #             break
        if not target_cell:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIPickUp",
                    "failure.default",
                    object=input_obj_des,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            find_obj = target_cell.color + " " + target_cell.type

        if target_cell.type != object_type:
            find_obj = target_cell.color + " " + target_cell.type
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIPickUp",
                    "failure.isBlocked",
                    object=find_obj,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        elif color and target_cell.color != color:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIPickUp",
                    "failure.isBlocked",
                    object=find_obj,
                    observation=self.env.get_observation(self.agent),
                ),
            )

        elif target_cell.can_pickup():
            object_type = target_cell.type
            if self.env.world.carrying is None:
                # if not self.agent.add_inventory(target_cell):
                #     return ActionFeedback(
                #         success=False,
                #         feedback=self.feedback_builder.build(
                #             "BabyAIPickUp",
                #             "failure.default",
                #             object=find_obj,
                #             observation=self.env.get_observation(self.agent),
                #         ),
                #     )
                # else:
                # self.agent.set_position(target_agent_pos)
                # self.env.world.carrying = target_cell
                # self.env.world.carrying.cur_pos = np.array([-1, -1])
                # self.env.world.grid.set(*target_pos, None)
                _, _, done, _, _ = self.env.world.step(self.env.world.actions.pickup)
                return BabyAIActionFeedback(
                    success=True,
                    task_success=done,
                    feedback=self.feedback_builder.build(
                        "BabyAIPickUp",
                        "success.default",
                        object=find_obj,
                        observation=self.env.get_observation(self.agent),
                        inventory=self.env.get_held_object_observation(),
                    ),
                )
            else:
                return BabyAIActionFeedback(
                    success=False,
                    task_success=False,
                    feedback=self.feedback_builder.build(
                        "BabyAIPickUp",
                        "failure.intentoryFilled",
                        object=find_obj,
                        observation=self.env.get_observation(self.agent),
                        inventory=self.env.get_held_object_observation(),
                    ),
                )
        else:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIPickUp",
                    "failure.default",
                    object=input_obj_des,
                    observation=self.env.get_observation(self.agent),
                ),
            )


@ACTION_REGISTERY.register()
class BabyAIDropAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)
        self.kwlist = ["object_type"]

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        if "object_type" in kwargs:
            object_type = kwargs.get("object_type")
        else:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
                ),
            )
        input_obj_des = object_type
        if "color" in kwargs:
            color = kwargs.get("color")
            input_obj_des = color + " " + object_type
        else:
            color = None

        fwd_pos = self.env.world.front_pos
        fwd_cell = self.env.world.grid.get(*fwd_pos)
        if self.env.world.carrying is None:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIDrop",
                    "failure.emptyInventory",
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            if (object_type != self.env.world.carrying.type) or (
                color and color != self.env.world.carrying.color
            ):
                return BabyAIActionFeedback(
                    success=False,
                    task_success=False,
                    feedback=self.feedback_builder.build(
                        "BabyAIDrop",
                        "failure.objectNotInInventory",
                        object=input_obj_des,
                        observation=self.env.get_observation(self.agent),
                        inventory=self.env.get_held_object_observation(),
                    ),
                )
            else:
                if not fwd_cell:
                    _, _, done, _, _ = self.env.world.step(self.env.world.actions.drop)
                    # self.env.world.grid.set(*fwd_pos, self.env.world.carrying)
                    # self.env.world.carrying.cur_pos = fwd_pos
                    # self.env.world.carrying = None
                    # self.agent.inventory.pop(0)
                    return BabyAIActionFeedback(
                        success=True,
                        task_success=done,
                        feedback=self.feedback_builder.build(
                            "BabyAIDrop",
                            "success.default",
                            object=input_obj_des,
                            observation=self.env.get_observation(self.agent),
                            inventory=self.env.get_held_object_observation(),
                        ),
                    )


@ACTION_REGISTERY.register()
class BabyAIToggleAction(BaseAction):
    def __init__(self, env, agent) -> None:
        super().__init__(env=env, agent=agent)

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        """
        Args:
            object_type: name of object
        """
        # fwd_pos = self.env.world.front_pos
        # fwd_cell = self.env.world.grid.get(*fwd_pos)
        if "object_type" in kwargs:
            object_type = kwargs.get("object_type")
        else:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "InvalidAction",
                    "failure.objectNotProvide",
                    observation=self.env.get_observation(self.agent),
                ),
            )

        input_obj_des = object_type
        if "color" in kwargs:
            color = kwargs.get("color")
            input_obj_des = color + " " + object_type
        else:
            color = None
        target_pos = self.env.world.front_pos
        target_cell = self.env.world.grid.get(*target_pos)
        # target_agent_pos = self.env.world.agent_pos
        # self.env.world.update_objs_poss()
        # step_count = 0
        # target_cell = None
        # for i in range(1, self.agent.view_size):
        #     next_pos = self.env.world.agent_pos + self.env.world.agent_dir * i
        #     if (
        #         next_pos[0] < self.env.world.width
        #         and next_pos[1] < self.env.world.height
        #     ):
        #         next_cell = self.env.world.grid.get(*next_pos)
        #         if next_cell is None or next_cell.can_overlap():
        #             step_count += 1
        #             target_agent_pos = next_pos
        #             continue
        #         else:
        #             target_cell = next_cell
        #             target_pos = next_pos
        #             break
        if not target_cell:
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIToggle",
                    "failure.default",
                    object=input_obj_des,
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            find_obj = target_cell.color + " " + target_cell.type

        if target_cell.type != object_type or (color and target_cell.color != color):
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=self.feedback_builder.build(
                    "BabyAIToggle",
                    "failure.isBlocked",
                    object=find_obj,
                    observation=self.env.get_observation(),
                ),
            )
        else:
            can_toggle = False
            # is_toggled = target_cell.toggle(self.env.world, target_pos)
            if target_cell.type == "door" and target_cell.is_locked:
                if (
                    isinstance(self.env.world.carrying, Key)
                    and self.env.world.carrying.color == target_cell.color
                ):
                    can_toggle = True
                else:
                    can_toggle = False
            elif target_cell.type == "door" and not target_cell.is_locked:
                can_toggle = True

            elif target_cell.type == "box":
                can_toggle = True

            if can_toggle:
                # self.agent.set_position(target_agent_pos)
                _, _, done, _, _ = self.env.world.step(self.env.world.actions.toggle)
                return BabyAIActionFeedback(
                    success=True,
                    task_success=False,
                    feedback=self.feedback_builder.build(
                        "BabyAIToggle",
                        "success.default",
                        object=find_obj,
                        observation=self.env.get_observation(self.agent),
                    ),
                )
            else:
                return BabyAIActionFeedback(
                    success=False,
                    task_success=False,
                    feedback=self.feedback_builder.build(
                        "BabyAIToggle",
                        "failure.default",
                        object=input_obj_des,
                        observation=self.env.get_observation(self.agent),
                    ),
                )
        # else:
        #     observed_objects = self.env.get_observed_objects()
        #     return ActionFeedback(
        #         success=False,
        #         feedback=self.feedback_builder.build(
        #             "BabyAIToggle", "failure.notInView", observation=observed_objects
        #         ),
        #     )
