from typing import Dict, Tuple
from attr import dataclass

from overrides import override
from langsuite.suit import (
    IllegalActionError,
    InvalidActionError,
    ParameterMissingError,
    UnexecutableWithAttrError,
)
from langsuite.suit import TaskAction,TaskActionWrapper
from langsuite.tasks.iqa_v0.message_handler import IQAStatus
from langsuite.worlds.basic2d_v0.actions import (
    BasicMove2D,
    BasicTurn2D,
    SwitchBoolAttr,
)
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0
from langsuite.worlds.basic2d_v0.physical_entity import (
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


# class CloseObject(TaskActionWrapper):
#     name: str = "close"

#     status_map: Dict[type, str] = {
#         IllegalActionError: "failure.notInView",
#         UnexecutableWithAttrError: "failure.notOpenable",
#         ParameterMissingError: "failure.objectNotProvide",
#     }

#     @property
#     @override
#     def wrapped_action(self):
#         return self._wrapped_action

#     def __init__(
#         self, agent: PhysicalAgent, world: Basic2DWorld_V0, object_id: str
#     ) -> None:
#         self._wrapped_action = SwitchBoolAttr(
#             agent=agent,
#             world=world,
#             object_id=object_id,
#             attr_name="isOpen",
#             expect_val=True,
#         )


@dataclass
class Stop(TaskAction):
    world: Basic2DWorld_V0
    agent: PhysicalAgent
    task_type: int
    target_status: IQAStatus
    answer: str
    name: str = "answer"

    def exec(self) -> Tuple[bool, dict]:
        info = dict()
        info["action"] = "Stop"
        if self.answer == self.target_status.answer:
            info["status"] = "success"
            info["reward"] = 1
        else:
            info["status"] = "failure"
            info["reward"] = 0
        return (True, info)
