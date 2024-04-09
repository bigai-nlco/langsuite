from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from overrides import EnforceOverrides, override


from langsuite.shapes import Line2D, Point2D
from langsuite.suit.exceptions import (
    IllegalActionError,
    InvalidActionError,
    ParameterMissingError,
    UnexecutableWithAttrError,
    UnexecutableWithSptialError,
    LimitExceededError,
)
from langsuite.suit import Action
from langsuite.utils import logging
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0, Object2D
from langsuite.worlds.basic2d_v0.physical_entity import LocateAt, PhysicalAgent


@dataclass
class InWorldAction(Action, EnforceOverrides):
    world: Basic2DWorld_V0
    agent: PhysicalAgent

    @override
    def make_info(self):
        info = super().make_info()
        for k, v in info.items():
            if isinstance(v, Object2D):
                info[k] = self.world.object_name2index(v.name)
        return info


class BasicMove2D(InWorldAction):
    def __init__(
        self, agent: PhysicalAgent, world: Basic2DWorld_V0, dist: Optional[float] = None
    ):
        super().__init__(world=world, agent=agent)
        if dist is None:
            self.distance = self.agent.step_size
        else:
            self.distance = dist
        self.dest = self.agent.rotation * self.distance + self.agent.position
        self.path = Line2D([self.agent.position, self.dest])

    @override
    def _executable_assert(self) -> None:
        super()._executable_assert()
        (has_intersects, intersected) = self.world.intersects(self.path)
        try:
            assert not has_intersects
        except AssertionError as e:
            raise IllegalActionError({"objects": intersected}) from e

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        self.agent.position = self.dest
        logging.logger.debug(f"move to {self.dest}")
        return (
            True,
            {"steps": self.distance / self.agent.step_size},
        )


@dataclass
class BasicTurn2D(InWorldAction):
    degree: Optional[float] = None

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        degree = self.degree or 90
        self.agent.rotation.rotate(degree)
        logging.logger.debug(f"turn to {self.agent.rotation}")
        return (True, {})


@dataclass
class InteractAction(InWorldAction):
    object_index: str
    object: Object2D = field(init=False)

    @override
    def _executable_assert(self):
        super()._executable_assert()
        try:
            self.object = self.world.get_object(self.object_index)
        except ParameterMissingError as e:
            raise e
        dist = self.world.distance_to_pos(self.agent, self.object)
        if not self.world.can_observe(self.agent, self.object):
            raise IllegalActionError({"object": self.object_index})
        if dist.modulus > self.agent.max_manipulate_distance:
            print(dist.modulus, self.agent.max_manipulate_distance)
            raise UnexecutableWithSptialError({"distance": dist.modulus})


class PickUp2D(InteractAction):
    @override
    def _executable_assert(self):
        try:
            super()._executable_assert()
        except IllegalActionError as e:
            if self.object in self.agent.inventory:
                raise InvalidActionError(
                    {
                        "object": self.object_index,
                        "inventory": ",".join(
                            map(
                                self.world.object_name2index,
                                map(lambda x: x.name, self.agent.inventory),
                            )
                        ),
                    }
                ) from e
            else:
                raise e
        if not hasattr(self.object, "isPickedUp"):
            raise UnexecutableWithAttrError({"attr": "isPickedUp"})
        if self.agent.inventory_capacity == len(self.agent.inventory):
            raise LimitExceededError(
                {
                    "capacity": self.agent.inventory_capacity,
                    "count": len(self.agent.inventory),
                }
            )

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        rel = LocateAt(self.agent, self.world.timestamp, Point2D(0, 0))
        # TODO calc location!
        self.object.update_position(rel)
        self.object.isPickedUp = True
        return (
            True,
            {
                "inventory": ",".join(
                    map(
                        self.world.object_name2index,
                        map(lambda x: x.name, self.agent.inventory),
                    )
                )
            },
        )


@dataclass
class Drop2D(InteractAction):
    rotate_degree: float = 45
    length: float = 0.5

    @override
    def _executable_assert(self):
        try:
            super()._executable_assert()
        except (UnexecutableWithSptialError, IllegalActionError) as e:
            logging.logger.debug(f"{self.object_index} {e.param_dict}")
            # Do not need to care the distance of an invertory
            pass
        if self.object._locate_at.receptacle != self.agent:
            raise UnexecutableWithSptialError(
                {"inventory": self.world.make_id_list(self.agent.inventory)}
            )

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        dist = deepcopy(self.agent.rotation)
        dist.rotate(self.rotate_degree)
        dist *= self.length
        rel = LocateAt(
            self.world.ground, self.world.timestamp, self.agent.position + dist
        )
        self.object.update_position(rel)
        self.object.isPickedUp = False
        return (True, {"inventory": self.world.make_id_list(self.agent.inventory)})


@dataclass
class Put2D(InteractAction):
    receptacle_index: str
    receptacle: Object2D = field(init=False)
    force: bool = True
    # Need to check object type if not force

    @override
    def _executable_assert(self):
        try:
            super()._executable_assert()
        except (UnexecutableWithSptialError, IllegalActionError) as e:
            logging.logger.debug(f"{self.object_index} {e.param_dict}")
            # Do not need to care the distance of an invertory
            pass
        if self.object._locate_at.receptacle != self.agent:
            raise UnexecutableWithSptialError(
                {"inventory": self.world.make_id_list(self.agent.inventory)}
            )
        try:
            self.receptacle = self.world.get_object(self.receptacle_index)
        except ParameterMissingError as e:
            raise e
        dist = self.world.distance_to_pos(self.agent, self.receptacle)
        if dist.modulus > self.agent.max_manipulate_distance:
            raise UnexecutableWithSptialError({"distance": dist.modulus})
        if not self.world.can_observe(self.agent, self.receptacle):
            raise IllegalActionError({"object": self.object_index})
        if not self.receptacle.receptacle:
            raise UnexecutableWithAttrError({"attr": "isReceptacle"})
        if getattr(self.receptacle, "isOpen", None) is False:
            raise UnexecutableWithAttrError({"attr": "isOpen"})

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        # TODO calc location!
        rel = LocateAt(self.receptacle, self.world.timestamp, Point2D(0, 0))
        self.object.update_position(rel)
        self.object.isPickedUp = False
        return (True, {})


@dataclass
class SwitchBoolAttr(InteractAction):
    attr_name: str
    expect_val: Optional[bool] = None
    premise_obj: Optional[str] = None

    def __post_init__(self):
        self.is_slice = self.attr_name == "isSliced"

    @override
    def _executable_assert(self):
        super()._executable_assert()
        if not (
            hasattr(self.object, self.attr_name)
            and isinstance(getattr(self.object, self.attr_name), bool)
        ):
            raise UnexecutableWithAttrError({"attr": self.attr_name})
        if self.premise_obj is not None and not any(
            # HACK XXX why endwith? this is a stupid hack for ButterKnife.
            getattr(x, "obj_type", "").endswith(self.premise_obj)
            for x in self.agent.inventory
        ):
            raise UnexecutableWithAttrError({"premise_obj": self.premise_obj})
        if (
            self.expect_val is not None
            and getattr(self.object, self.attr_name) != self.expect_val
        ):
            raise InvalidActionError({"value": self.expect_val})

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        switched = not getattr(self.object, self.attr_name)
        self.object.update_attr(self.attr_name, switched, None)
        return (True, {})


@dataclass
class SetBoolAttrWith(InteractAction):
    # It is not receptacle, just an extra object. this is for automatic init, don't change the name.
    receptacle_index: str
    receptacle: Object2D = field(init=False)

    @property
    @abstractmethod
    def attr_name(self) -> str:
        pass

    @property
    @abstractmethod
    def expect_val(self) -> Any:
        pass

    @override
    def _executable_assert(self):
        super()._executable_assert()
        if not (
            hasattr(self.object, self.attr_name)
            and isinstance(getattr(self.object, self.attr_name), bool)
        ):
            raise UnexecutableWithAttrError({"attr": self.attr_name})
        if not (
            self.expect_val is not None
            and getattr(self.object, self.attr_name) != self.expect_val
        ):
            raise InvalidActionError({"value": self.expect_val})

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        self.object.update_attr(self.attr_name, self.expect_val, None)
        return (True, {})


class Clean(SetBoolAttrWith):
    @property
    @override
    def attr_name(self) -> str:
        return "isDirty"

    @property
    @override
    def expect_val(self) -> bool:
        return False

    @override
    def _executable_assert(self):
        super()._executable_assert()
        if not (
            self.receptacle.obj_type
            in {"Faucet", "SinkBasin", "BathhubBasin", "ShowerHead"}
            or getattr(self.receptacle, "isFilledWithLiquid") == "Water"
        ):
            raise UnexecutableWithAttrError({"premise_obj": self.receptacle.obj_type})


class Cook(SetBoolAttrWith):
    @property
    @override
    def attr_name(self) -> str:
        return "isCooked"

    @property
    @override
    def expect_val(self) -> bool:
        return True

    @override
    def _executable_assert(self):
        super()._executable_assert()
        if not self.receptacle.obj_type in {"StoveBurner", "Microwave"}:
            raise UnexecutableWithAttrError({"premise_obj": self.receptacle.obj_type})

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        result = super()._exec()
        if getattr(self.object, "temperature"):
            self.object.update_attr("temperature", "Hot", None)
        return result


class Heat(SetBoolAttrWith):
    @property
    @override
    def attr_name(self) -> str:
        return "temperature"

    @property
    @override
    def expect_val(self) -> str:
        return "Hot"

    @override
    def _executable_assert(self):
        super()._executable_assert()
        if not (
            self.receptacle.obj_type in {"StoveBurner", "Microwave"}
            or getattr(self.receptacle, "isHeatSource")
        ):
            raise UnexecutableWithAttrError({"premise_obj": self.receptacle.obj_type})

    @override
    def _exec(self) -> Tuple[bool, Dict[str, object]]:
        result = super()._exec()
        if getattr(self.object, "isCooked"):
            self.object.update_attr("isCooked", True, None)
        return result


class Cool(SetBoolAttrWith):
    @property
    @override
    def attr_name(self) -> str:
        return "temperature"

    @property
    @override
    def expect_val(self) -> str:
        return "Cold"

    @override
    def _executable_assert(self):
        super()._executable_assert()
        if not (
            self.receptacle.obj_type in {"Fridge"}
            or getattr(self.receptacle, "isColdSource")
        ):
            raise UnexecutableWithAttrError({"premise_obj": self.receptacle.obj_type})

# class GoTo(InteractAction):
#     @override
#     def _exec(self) -> Tuple[bool, Dict[str, object]]:
#         # FIXME only work with single room
#         it = iter(self.world.rooms.values())
#         next(it)
#         room = next(it)
#         grid_world = GridWorld(room.geometry, self.agent.step_size)
#         for x in self.world._objects.values():
#             grid_world.add_object(x)
#         grid_world.get_path(self.agent.position, self.object.position)

#         return (
#             True,
#             {
#                 "inventory": ",".join(
#                     map(
#                         self.world.object_name2index,
#                         map(lambda x: x.name, self.agent.inventory),
#                     )
#                 )
#             },
#         )


