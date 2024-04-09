from abc import ABC, abstractmethod
import copy
import random
import re
from typing import Dict, Set
from dataclasses import dataclass
from overrides import EnforceOverrides, override
from plotly.graph_objects import Figure, Scatter
from langsuite.constants import CSS4_COLORS
from langsuite.utils import math_utils
from langsuite.utils.logging import logger
import numpy as np
import langsuite.worlds.basic2d_v0.utils as WUtils
from langsuite.shapes import Cone2D, Geometry, Line2D, Point2D, Polygon2D, Vector2D
from shapely import transform, Polygon, Point


class PhysicalEntity2D(ABC, EnforceOverrides):
    def __init__(self) -> None:
        self.inventory: Set["PhysicalEntity2D"] = set()

    def add_to_inventory(self, obj: "PhysicalEntity2D"):
        self.inventory.add(obj)

    def remove_from_inventory(self, obj: "PhysicalEntity2D"):
        self.inventory.remove(obj)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def geometry(self) -> Geometry:
        pass

    @property
    @abstractmethod
    def position(self) -> Point2D:
        pass

    def intersects(self, other_shape: Geometry) -> bool:
        return self.geometry.shapely_geo.intersects(other_shape.shapely_geo)

    @abstractmethod
    def list_textual_attrs(self) -> Dict[str, object]:
        pass

    @abstractmethod
    def render(self, fig: Figure, label: str):
        pass


@dataclass(frozen=True)
class LocateAt:
    receptacle: PhysicalEntity2D
    timestamp: int
    distance: Point2D


@dataclass
class Wall2D:
    geometry: Line2D

    @classmethod
    def create_by_room(cls, wall_type, **kwargs):
        if wall_type == "WALL":
            return type("Wall2D", (), {"position": kwargs["position"]})
        # TODO: door and window

    def intersects(self, other_shape: Geometry) -> bool:
        return self.geometry.shapely_geo.intersects(other_shape.shapely_geo)


class Room2D(PhysicalEntity2D):
    def __init__(self, room_id: str, geo_shape: Polygon) -> None:
        super().__init__()
        self.room_id = room_id
        self._geometry: Polygon2D = Polygon2D(geo_shape)
        self._position: Point2D = self._geometry.centroid
        coords = self._geometry.coords
        length = len(coords)
        self._walls = [
            Wall2D(Line2D([coords[i], coords[(i + 1) % length]])) for i in range(length)
        ]

    @property
    def name(self):
        return self.room_id

    # TODO
    def set_door(self, **kwargs):
        pass

    # TODO
    def set_window(self, **kwargs):
        pass

    @property
    def geometry(self) -> Polygon2D:
        return self._geometry

    @property
    def position(self) -> Point2D:
        return self._position

    @override
    def intersects(self, other_shape: Geometry) -> bool:
        return any(x.intersects(other_shape) for x in self._walls)

    @override
    def list_textual_attrs(self) -> Dict[str, object]:
        return {"type": "Room"}

    @override
    def render(self, fig: Figure, label: str):
        x, y = self.geometry.shapely_geo.exterior.xy
        fig.add_trace(
            Scatter(
                x=np.array(x),
                y=np.array(y),
                fill="toself",
                fillcolor="aliceblue",
                name=label,
                line=dict(color="gray"),
            )
        )


class Object2D(PhysicalEntity2D):
    colorscales = list(CSS4_COLORS.keys())
    color_registry = dict()

    __slots__ = [
        "isOpen",
        "isPickedUp",
        "isToggled",
        "isFilledWithLiquid",
        "isSliced",
        "isCooked",
        "isBroken",
        "isDirty",
        "isUsedUp",
        "isHeatSource",
        "isColdSource",
        "temperature",
        "receptacle",
    ]

    def _read_attrs(
        self, data: dict, rules: dict = WUtils.DataLoader.get_object_rules()
    ):
        for attr_rule in rules["attr"]:
            attr_name = attr_rule["name"]
            if "premise" in attr_rule:
                if data.get(attr_rule["premise"]):
                    setattr(self, attr_name, data.get(attr_name))
            else:
                setattr(self, attr_name, data.get(attr_name))
        for func_rule in rules["receptacle_function"]:
            source_domain = func_rule.get("source_domain")
            if (
                ((not source_domain["type"]) or source_domain["type"] == self.obj_type)
                and all(
                    getattr(self, attr_name) == val
                    for attr_name, val in source_domain["has_attr"].items()
                )
                and (
                    (not func_rule["premise_attr"])
                    or hasattr(self, func_rule["premise_attr"])
                )
            ):
                self._receptacle_function.append(func_rule)
        for func_rule in rules["inventory_function"]:
            source_domain = func_rule.get("source_domain", None)
            if (
                ((not source_domain["type"]) or source_domain["type"] == self.obj_type)
                and all(hasattr(self, attr) for attr in source_domain["has_attr"])
                and (
                    (not func_rule["premise_attr"])
                    or hasattr(self, func_rule["premise_attr"])
                )
            ):
                self._inventory_function.append(func_rule)

    def __init__(
        self,
        obj_id: str,
        obj_type: str,
        rotation: Point2D,
        geometry: Polygon,
        locate_at: LocateAt,
        **kwargs,
    ):
        super().__init__()
        self.obj_name = obj_id
        self.obj_type = obj_type
        self.rotation = rotation
        if self.obj_type not in self.color_registry:
            select_color = random.choice(self.colorscales)
            self.color_registry.update({self.obj_type: select_color})
            self.colorscales.remove(select_color)

        self._locate_at = locate_at
        locate_at.receptacle.inventory.add(self)
        self._position = self._locate_at.receptacle.position + self._locate_at.distance
        self._shape = Polygon2D(
            transform(geometry, lambda x: x - [self._position.x, self._position.y])
        )
        self._geometry = Polygon2D(geometry)
        try:
            assert self._shape.shapely_geo.buffer(1e-6).contains(Point(0, 0))
        except AssertionError as e:
            logger.error(
                "%s: %s. \n\tGeo: %s\n\tShape: %s\n\tPosition: %s",
                self.name,
                e,
                geometry,
                self._shape,
                self._position,
            )
            raise e

        self._inventory_function: list = list()
        self._receptacle_function: list = list()
        if "attrs" in kwargs:
            self._read_attrs(kwargs["attrs"])
        self._timestamp: int = locate_at.timestamp

    def copy(self):
        new_obj = copy.copy(self)

        new_obj.inventory = set(self.inventory)

        new_obj._position = copy.deepcopy(self._position)
        new_obj._geometry = copy.deepcopy(self.geometry)
        new_obj._shape = copy.deepcopy(self._shape)

        return new_obj

    def _update_location(self):
        self._position = self._locate_at.receptacle.position + self._locate_at.distance
        self._geometry = Polygon2D(
            list(map(lambda x: x + self._position, self._shape.coords))
        )
        self._timestamp = self._locate_at.timestamp

    @override
    def add_to_inventory(self, obj: "Object2D"):
        self.inventory.add(obj)
        for f in self._receptacle_function:
            attr = f["premise_attr"]
            if attr is not None and getattr(self, attr) != f.get("premise_value", True):
                continue
            if hasattr(obj, f["on_attr"]):
                if "value" in f:
                    setattr(obj, f["on_attr"], f["value"])
                else:
                    setattr(obj, f["on_attr"], getattr(self, f["on_attr"]))

    @property
    def name(self):
        return self.obj_name

    @property
    def geometry(self):
        if self._locate_at.timestamp > self._timestamp:
            self._update_location()
        return self._geometry

    @property
    def position(self) -> Point2D:
        if self._locate_at.timestamp > self._timestamp:
            self._update_location()
        return self._position

    def update_position(self, locate_at: LocateAt):
        self._locate_at.receptacle.remove_from_inventory(self)
        self._locate_at = locate_at
        new_receptacle = self._locate_at.receptacle
        new_receptacle.add_to_inventory(self)
        for f in self._inventory_function:
            assert "premise_attr" in f
            if getattr(self, f["premise_attr"]) != f.get("premise_value", True):
                continue
            if hasattr(new_receptacle, f["on_attr"]):
                if "value" in f:
                    setattr(new_receptacle, f["on_attr"], f["value"])
                else:
                    setattr(new_receptacle, f["on_attr"], getattr(self, f["on_attr"]))

    def update_attr(self, attr_name: str, val: object):
        assert hasattr(self, attr_name)
        # I don't think the data quality is good enough to do such check
        # old_val = getattr(self, attr_name)
        # assert old_val is None or isinstance(val, type(old_val))
        logger.debug(f"Set {attr_name} of {self.name} as {val}")
        setattr(self, attr_name, val)

        for f in self._receptacle_function:
            if f["premise_attr"] != attr_name or f.get("premise_value", True) != val:
                continue
            for obj in self.inventory:
                if hasattr(obj, f["on_attr"]):
                    if "value" in f:
                        setattr(obj, f["on_attr"], f["value"])
                    else:
                        setattr(obj, f["on_attr"], getattr(self, f["on_attr"]))

        for f in self._inventory_function:
            if f["premise_attr"] != attr_name or f.get("premise_value", True) != val:
                continue
            if hasattr(self._locate_at.receptacle, f["on_attr"]):
                if "value" in f:
                    setattr(self._locate_at.receptacle, f["on_attr"], f["value"])
                else:
                    setattr(
                        self._locate_at.receptacle,
                        f["on_attr"],
                        getattr(self, f["on_attr"]),
                    )

    @override
    def list_textual_attrs(self) -> Dict[str, object]:
        collected: Dict[str, object] = {"type": self.obj_type}
        for key in self.__slots__:
            if hasattr(self, key) and getattr(self, key) is not None:
                collected[key] = getattr(self, key)
        return collected

    @override
    def render(self, fig: Figure, label: str):
        x, y = self.geometry.shapely_geo.exterior.xy
        fig.add_trace(
            Scatter(
                x=np.array(x),
                y=np.array(y),
                fill="toself",
                fillcolor=self.color_registry.get(self.obj_type),
                name=f"{label} ({self.name})",
                mode="lines",
                line=dict(width=0),
            )
        )


class PhysicalAgent(PhysicalEntity2D):
    def __init__(
        self,
        agent_id: str,
        position: Point2D,
        rotation: Vector2D,
        step_size: float,
        focal_length: float,
        max_view_distance: float,
        max_manipulate_distance: float,
        inventory_capacity: int,
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self._position = position
        self.rotation = rotation
        self.step_size = step_size
        self.focal_length = focal_length
        self.view_angle = math_utils.compute_horizonal_aov(focal_length)
        self.max_view_distance = max_view_distance
        self.max_manipulate_distance = max_manipulate_distance
        self.inventory_capacity = inventory_capacity
        self.max_view_steps = self.max_view_distance / self.step_size
        self.left_degree = self.view_angle / 2
        self.right_degree = self.view_angle / 2

    def update_config(self, config):
        if "focal_length" in config:
            self.focal_length = config["focal_length"]
            self.view_angle = math_utils.compute_horizonal_aov(config["focal_length"])
        if "step_size" in config:
            self.step_size = config["step_size"]
        if "max_view_distance" in config:
            self.max_view_distance = config["max_view_distance"]
            self.max_view_steps = self.max_view_distance / self.step_size
            # TODO XXX they should be set seperately?
            self.max_manipulate_distance = config["max_view_distance"]

    @property
    def name(self):
        return self.agent_id

    @property
    def geometry(self):
        return self._position

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._position = new_position

    @property
    def view_geometry(self):
        return Cone2D(
            center=self.position,
            radius=self.max_view_distance,
            direction=self.rotation,
            angle=self.view_angle,
        )

    @override
    def list_textual_attrs(self) -> Dict[str, object]:
        return {"type": "Agent"}

    @override
    def render(self, fig: Figure, label: str):
        radius = 0.05
        svg = re.findall(r'd="(.*?)"', self.view_geometry.shapely_geo.svg())[0]
        fig.add_shape(
            type="path", path=svg, fillcolor="orange", opacity=0.2, line=dict(width=0)
        )
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=self.position.x - radius,
            y0=self.position.y - radius,
            x1=self.position.x + radius,
            y1=self.position.y + radius,
            fillcolor="orange",
            name=label,  # TODO Why this not work?
            line=dict(width=0),
        )
