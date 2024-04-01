from typing import ClassVar, Dict, Generator, Tuple, Any
from attr import dataclass

from overrides import override
from langsuite.suit import (
    IllegalActionError,
    InvalidActionError,
    LimitExceededError,
    ParameterMissingError,
    UnexecutableWithAttrError,
    UnexecutableWithSptialError,
)
from langsuite.suit import TaskAction,TaskActionWrapper
from langsuite.tasks.alfred_v0.message_handler import AlfredStatus
from langsuite.utils import logging
from langsuite.worlds.basic2d_v0.actions import (
    BasicMove2D,
    BasicTurn2D,
    Drop2D,
    PickUp2D,
    Put2D,
    SwitchBoolAttr,
)
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0
from langsuite.worlds.basic2d_v0.physical_entity import (
    Object2D,
    PhysicalAgent,
)


class MoveAhead(TaskActionWrapper):
    name: str = "move_ahead"
    
    status_map: Dict[type, str] = {IllegalActionError: "failure.isBlocked"}

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(self, agent: PhysicalAgent, world: Basic2DWorld_V0):
        self._wrapped_action = BasicMove2D(
            agent=agent, world=world, dist=agent.step_size
        )


class RotateLeft(TaskActionWrapper):
    name: str = "turn_left"

    status_map: Dict[type, str] = {}

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(self, agent: PhysicalAgent, world: Basic2DWorld_V0):
        self._wrapped_action = BasicTurn2D(agent=agent, world=world, degree=-90)

    @override
    def _post_process(self, info: dict):
        super()._post_process(info)
        info["degree"] = -info["degree"]


class RotateRight(TaskActionWrapper):
    name: str = "turn_right"    
    
    status_map: Dict[type, str] = {}

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(self, agent: PhysicalAgent, world: Basic2DWorld_V0):
        self._wrapped_action = BasicTurn2D(agent=agent, world=world, degree=90)


class PickupObject(TaskActionWrapper):
    name: str = "pick_up"

    _status_map = {
        IllegalActionError: "failure.notInView",
        LimitExceededError: "failure.intentoryFilled",
        ParameterMissingError: "failure.objectNotProvide",
        InvalidActionError: "success.default"
    }

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
    ) -> None:
        self._wrapped_action = PickUp2D(agent=agent, world=world, object_id=object_id)

    @property
    @override
    def status_map(self):
        return self._status_map


class DropObject(TaskActionWrapper):
    name: str = "drop"

    status_map: Dict[type, str] = {
        UnexecutableWithSptialError: "failure.objectNotInInventory",
    }

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
    ) -> None:
        self._wrapped_action = Drop2D(agent=agent, world=world, object_id=object_id)

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

class PutObject(TaskActionWrapper):
    name: str = "put"

    status_map: Dict[type, str] = {
        IllegalActionError: "failure.notInView",
        UnexecutableWithSptialError: "failure.NotInInventory",
    }

    def __init__(
        self,
        agent: PhysicalAgent,
        world: Basic2DWorld_V0,
        object_id: str,
        receptacle_id: str,
    ) -> None:
        self._wrapped_action = Put2D(
            agent=agent,
            world=world,
            object_id=object_id,
            receptacle_id=receptacle_id,
            force=True,
        )

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

class OpenObject(TaskActionWrapper):
    name: str = "open"

    _status_map = {
        IllegalActionError: "failure.notInView",
        UnexecutableWithAttrError: "failure.notOpenable",
        ParameterMissingError: "failure.objectNotProvide",
        # HACK
        InvalidActionError: "success.default",
    }

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
    ) -> None:
        self._wrapped_action = SwitchBoolAttr(
            agent=agent,
            world=world,
            object_id=object_id,
            attr_name="isOpen",
            expect_val=False,
        )

    @property
    @override
    def status_map(self):
        return self._status_map


class CloseObject(TaskActionWrapper):
    name: str = "close"

    status_map: Dict[type, str] = {
        IllegalActionError: "failure.notInView",
        UnexecutableWithAttrError: "failure.notOpenable",
        ParameterMissingError: "failure.objectNotProvide",
    }

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
    ) -> None:
        self._wrapped_action = SwitchBoolAttr(
            agent=agent,
            world=world,
            object_id=object_id,
            attr_name="isOpen",
            expect_val=True,
        )

class ToggleObjectOn(TaskActionWrapper):
    name: str = "toggle_on"

    status_map: Dict[type, str] = {
        IllegalActionError: "failure.notInView",
        UnexecutableWithAttrError: "failure.notToggleable",
    }

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
    ) -> None:
        self._wrapped_action = SwitchBoolAttr(
            agent=agent,
            world=world,
            object_id=object_id,
            attr_name="isToggled",
            expect_val=False,
        )


class ToggleObjectOff(TaskActionWrapper):
    name: str = "toggle_off"

    status_map: Dict[type, str] = {
        IllegalActionError: "failure.notInView",
        UnexecutableWithAttrError: "failure.notToggleable",
    }

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
    ) -> None:
        self._wrapped_action = SwitchBoolAttr(
            agent=agent,
            world=world,
            object_id=object_id,
            attr_name="isToggled",
            expect_val=True,
        )


class SliceObject(TaskActionWrapper):
    name: str = "slice"
    is_slice = True

    status_map: Dict[type, str] = {
        IllegalActionError: "failure.notInView",
        UnexecutableWithAttrError: "failure.notSliceable",
        InvalidActionError: "failure.objectIsSliced",
    }

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
    ) -> None:
        self._wrapped_action = SwitchBoolAttr(
            agent=agent,
            world=world,
            object_id=object_id,
            attr_name="isSliced",
            expect_val=False,
            premise_obj="Knife"
        )

@dataclass
class Stop(TaskAction):   
    world: Basic2DWorld_V0
    agent: PhysicalAgent
    task_type: str
    target_status: AlfredStatus
    answer: Any
    name: ClassVar[str] = "Stop"

    def look_at_obj_in_light(self) -> bool:
        look_at = any(
            isinstance(obj, Object2D)
            and obj.obj_type == self.target_status.object_target
            for obj in self.agent.inventory
        )
        in_light = self.world.exists_in_obs(
            self.agent, self.target_status.toggle_target, "isToggled", True
        )
        logging.logger.debug("look_at:%s in_light:%s", look_at, in_light)
        return look_at and in_light

    def _placed_at(
        self, obj: Object2D, obj_type: str, rec_type: str, need_slice: bool
    ) -> bool:
        self_type_match = obj.obj_type == obj_type
        recept_type_match = (
            getattr(obj._locate_at.receptacle, "obj_type", None) == rec_type
        )
        slice_match = (not need_slice) or getattr(obj, "isSliced", False) is True
        if (self_type_match and recept_type_match and slice_match):
            logging.logger.debug(
                "Matched with %s", obj.name
            )
        return self_type_match and recept_type_match and slice_match

    def pick_and_place_simple(self) -> bool:
        return any(
            self._placed_at(
                obj,
                self.target_status.object_target,
                self.target_status.parent_target,
                self.target_status.object_sliced,
            )
            for obj in self.world._objects.values()
        )

    def pick_two_obj_and_place(self) -> bool:
        def at_least_two(x: Generator):
            return any(x) and any(x)
        return at_least_two(
            self._placed_at(
                obj,
                self.target_status.object_target,
                self.target_status.parent_target,
                self.target_status.object_sliced,
            )
            for obj in self.world._objects.values()
        )            

    def pick_and_place_with_movable_recep(self) -> bool:
        def _placed_with_mrecep(obj: Object2D):
            placed_at = self._placed_at(
                obj,
                self.target_status.object_target,
                self.target_status.mrecep_target,
                self.target_status.object_sliced,
            )
            if not placed_at:
                return False
            mrecep = obj._locate_at.receptacle
            mrecep_at = (
                isinstance(mrecep, Object2D)
                and getattr(mrecep._locate_at.receptacle, "obj_type", None)
                == self.target_status.parent_target
            )
            return mrecep_at

        return any(_placed_with_mrecep(obj) for obj in self.world._objects.values())

    def pick_heat_then_place_in_recep(self) -> bool:
        return any(
            self._placed_at(
                obj,
                self.target_status.object_target,
                self.target_status.parent_target,
                self.target_status.object_sliced,
            )
            and (getattr(obj, "temperature", None) == "Hot")
            for obj in self.world._objects.values()
        )

    def pick_cool_then_place_in_recep(self) -> bool:
        return any(
            self._placed_at(
                obj,
                self.target_status.object_target,
                self.target_status.parent_target,
                self.target_status.object_sliced,
            )
            and (getattr(obj, "temperature", None) == "Cold")
            for obj in self.world._objects.values()
        )

    def pick_clean_then_place_in_recep(self) -> bool:
        return any(
            self._placed_at(
                obj,
                self.target_status.object_target,
                self.target_status.parent_target,
                self.target_status.object_sliced,
            )
            and (
                getattr(obj, "isDirty", None) is False
            )  # FIXME will find obj that not been cleaned but wasn't dirty.
            for obj in self.world._objects.values()
        )

    def exec(self) -> Tuple[bool, dict]:
        info = dict()
        func = getattr(self, self.task_type)
        info["action"] = "Stop"
        if func():
            info["status"] = "success"
            info["reward"] = 1
        else:
            info["status"] = "failure"
            info["reward"] = 0
        return (True, info)
