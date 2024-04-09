from typing import Dict, Tuple
from attr import dataclass

from overrides import override
from langsuite.suit import (
    IllegalActionError,
    InvalidActionError,
    ParameterMissingError,
    UnexecutableWithAttrError,
)
from langsuite.suit import TaskAction, TaskActionWrapper
from langsuite.suit.exceptions import UnexecutableWithSptialError
from langsuite.tasks.rearrange_v0.message_handler import RearrangeStatus
from langsuite.tasks.rearrange_v0.utils import compute_square_hack
from langsuite.worlds.basic2d_v0.actions import (
    BasicMove2D,
    BasicTurn2D,
    Drop2D,
    Put2D,
    SwitchBoolAttr,
)
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0
from langsuite.worlds.basic2d_v0.physical_entity import (
    PhysicalAgent,
    Room2D,
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
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_index: str
    ) -> None:
        self._wrapped_action = SwitchBoolAttr(
            agent=agent,
            world=world,
            object_index=object_index,
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
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_index: str
    ) -> None:
        self._wrapped_action = SwitchBoolAttr(
            agent=agent,
            world=world,
            object_index=object_index,
            attr_name="isOpen",
            expect_val=True,
        )


class DropObject(TaskActionWrapper):
    name: str = "drop"

    status_map: Dict[type, str] = {
        UnexecutableWithSptialError: "failure.objectNotInInventory",
    }

    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_index: str
    ) -> None:
        self._wrapped_action = Drop2D(agent=agent, world=world, object_index=object_index)

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
        object_index: str,
        receptacle_index: str,
    ) -> None:
        self._wrapped_action = Put2D(
            agent=agent,
            world=world,
            object_index=object_index,
            receptacle_index=receptacle_index,
            force=True,
        )

    @property
    @override
    def wrapped_action(self):
        return self._wrapped_action


@dataclass
class Stop(TaskAction):
    world: Basic2DWorld_V0
    agent: PhysicalAgent
    task_type: int
    target_status: RearrangeStatus
    answer: str
    name: str = "answer"

    def _location_match(self, rec_type: type, cur_rec, tar_rec, cur_pos, tar_pos):
        if rec_type == Room2D:
            return cur_pos == tar_pos
        else:
            return cur_rec == tar_rec

    def compute(self):
        change_rec = self.target_status.change_rec
        change_open = self.target_status.change_open
        change_pos = self.target_status.change_pos

        arranged_set = (
            set(change_rec.keys()) | set(change_open.keys()) | set(change_pos.keys())
        )
        fixed = 0
        misplaced = 0
        no_fixed = False

        for obj_name in self.target_status.target_pos:

            # HACK only work for single room
            it = iter(self.world.rooms.values())
            next(it)  # drop ground
            room_name = next(it).name

            obj = self.world._objects[obj_name]
            tar_rec = self.target_status.target_rec.get(obj_name, room_name)
            cur_rec = obj._locate_at.receptacle.name
            rec_type = obj._locate_at.receptacle.__class__
            tar_pos = self.target_status.target_pos[obj_name]
            cur_pos = compute_square_hack(self.world, obj.position)
            tar_open = self.target_status.target_open[obj_name]
            cur_open = getattr(obj, "isOpen", None)

            if cur_open != tar_open or not self._location_match(
                rec_type, cur_rec, tar_rec, cur_pos, tar_pos
            ):
                misplaced += 1
                if obj_name in change_rec:
                    if cur_rec != change_rec[obj_name]:
                        no_fixed = True
                elif obj_name in change_open:
                    if cur_open != change_open[obj_name]:
                        no_fixed = True
                elif obj_name in change_pos:
                    if cur_pos != change_pos[obj_name]:
                        no_fixed = True
            elif obj_name in arranged_set:
                fixed += 1

        if no_fixed:
            fixed = 0

        return len(arranged_set), fixed, misplaced

    def exec(self) -> Tuple[bool, dict]:
        info = dict()
        info["action"] = "Stop"
        arranged, fixed, misplaced = self.compute()
        if fixed == arranged:
            info["status"] = "success"
            info["reward"] = 1
        else:
            info["status"] = "failure"
            info["reward"] = 0
        info["fixed"] = 1.0 * fixed / arranged
        info["misplaced"] = 1.0 * misplaced / arranged

        print(info)
        return (True, info)
