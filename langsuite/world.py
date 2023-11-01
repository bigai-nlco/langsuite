# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Dict, Optional, Tuple, Union

from langsuite.shapes import Geometry, Polygon2D
from langsuite.utils.registry import Registry


class ObjectType(Enum):
    OBJECT = 1
    ROOM = 2
    WALL = 3
    WINDOW = 4
    DOOR = 5


class Object2D:
    def __init__(
        self,
        obj_type: ObjectType,
        id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Geometry] = None,
        asset_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.id = id
        self.asset_id = asset_id
        self.alias = alias
        self.obj_type = obj_type
        self.geometry = geometry
        self.props = dict()
        for k, val in kwargs.items():
            self.props[k] = val

        self.walls = defaultdict()
        self.doors = defaultdict()
        self.windows = defaultdict()
        if "children" in self.props:
            self.children = self.props["children"]
        else:
            self.children = defaultdict()
        self.chilren_types = [ObjectType.OBJECT]

    @classmethod
    def create(cls, obj_data):
        return NotImplementedError()

    def __repr__(self) -> str:
        obj_string = f"asset_id: {self.asset_id}"
        return obj_string

    def contains(self, other) -> bool:
        """Returns True is another object is in current object

        Args:
            other: Object2D: an object instance
        """
        if not isinstance(other, Object2D):
            return ValueError(
                f"Invalid input: other has to be of type Object ({type(other)} given)"
            )
        if other.obj_type not in self.chilren_types:
            return False
        if other.obj_type == ObjectType.WALL:
            return other.id in self.walls.keys()
        elif other.obj_type == ObjectType.DOOR:
            return other.id in self.doors.keys()
        elif other.obj_type == ObjectType.WINDOW:
            return other.id in self.windows.keys()
        elif other.obj_type == ObjectType.OBJECT:
            return other.id in self.children.keys()
        else:
            raise ValueError(f"Invalid input: {type(other)}.")

    def add_wall(self, wall) -> Optional[str]:
        if ObjectType.WALL not in self.chilren_types:
            raise ValueError(f"Unable to add type {wall.obj_type}")
        if wall.id in self.wall:
            return wall.id
        self.walls[wall.id] = wall
        return wall.id

    def add_door(self, door) -> Optional[str]:
        if ObjectType.DOOR not in self.chilren_types:
            raise ValueError(f"Unable to add type {door.obj_type}")
        if door.id in self.doors:
            return door.id
        self.doors[door.id] = door
        return door.id

    def add_window(self, window) -> Optional[str]:
        if ObjectType.WINDOW not in self.chilren_types:
            raise ValueError(f"Unable to add type {window.obj_type}")

        if window.id in self.windows:
            return window.id
        self.windows[window.id] = window
        return window.id

    def add_object(self, object) -> Optional[str]:
        if ObjectType.OBJECT not in self.chilren_types:
            raise ValueError(f"Unable to add type {object.obj_type}")

        if object.id in self.children:
            return object.id
        self.children[object.id] = object
        return object.id

    def update_position(self, position):
        diff = position - self.position
        coords = []
        for i in range(len(self.geometry.coords)):
            coords.append(self.geometry.coords[i] + diff)
        self.geometry = Polygon2D(coords)
        self.position = position


class Door(Object2D):
    def __init__(
        self,
        door_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        asset_id: Optional[str] = None,
        room2room: Tuple[str] = [],
        openable: bool = True,
        is_open: bool = True,
        **kwargs,
    ):
        super().__init__(
            ObjectType.DOOR,
            door_id,
            alias=alias,
            geometry=geometry,
            asset_id=asset_id,
            **kwargs,
        )
        self.room2room = room2room
        self.openable = openable
        self.is_open = is_open
        self.wall = None
        self.chilren_types = []


class Window(Object2D):
    def __init__(
        self,
        window_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        asset_id: Optional[str] = None,
        room2room: Tuple[str] = [],
        **kwargs,
    ):
        super().__init__(
            ObjectType.WINDOW,
            window_id,
            alias=alias,
            geometry=geometry,
            asset_id=asset_id,
            **kwargs,
        )
        self.room2room = room2room
        self.chilren_types = []


class Wall(Object2D):
    def __init__(
        self,
        wall_id: str,
        *,
        alias: Optional[str],
        geometry: Optional[Geometry],
        asset_id: Optional[str],
        room2room: Union[Tuple[str], str] = [],
        **kwargs,
    ):
        super().__init__(
            ObjectType.WALL,
            wall_id,
            alias=alias,
            geometry=geometry,
            asset_id=asset_id,
            **kwargs,
        )
        self.chilren_types = [ObjectType.OBJECT, ObjectType.DOOR, ObjectType.WINDOW]
        self.room2room = [room2room] if type(room2room) == str else room2room


class Room(Object2D):
    def __init__(
        self,
        room_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        asset_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            ObjectType.ROOM,
            room_id,
            alias=alias,
            geometry=geometry,
            asset_id=asset_id,
            **kwargs,
        )
        self.chilren_types = [
            ObjectType.OBJECT,
            ObjectType.DOOR,
            ObjectType.WINDOW,
            ObjectType.WALL,
        ]


WORLD_REGISTRY = Registry("world")


class World:
    def __init__(self, world_id: str):
        self.world_id = world_id
        self.rooms: Dict[str, Room] = dict()
        self.walls: Dict[str, Wall] = dict()
        self.doors: Dict[str, Door] = dict()
        self.windows: Dict[str, Window] = dict()
        self.objects: Dict[str, Object2D] = dict()
        self.grid_size = None
        self.room_polygons = None
        self.id2object = {}

    @classmethod
    def create(cls, world_cfg):
        world_type = world_cfg.get("type")
        if world_type is None or len(world_type) == 0:
            raise ValueError("World type must be provided to create a world.")

        if WORLD_REGISTRY.hasRegistered(world_type):
            return WORLD_REGISTRY.get(world_type).create(world_cfg)
        else:
            raise NotImplementedError(f"World type {world_type} not found.")

    def add_room(self, room: Room) -> Optional[str]:
        return NotImplementedError()
