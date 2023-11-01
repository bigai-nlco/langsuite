# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from langsuite.constants import CSS4_COLORS
from langsuite.shapes import Geometry, Point2D, Polygon2D
from langsuite.utils.logging import logger
from langsuite.world import (
    WORLD_REGISTRY,
    Door,
    Object2D,
    ObjectType,
    Room,
    Wall,
    Window,
    World,
)

CwahPath = Path(__file__).parent


def ToEulerAngles(q):
    sinp = 2 * (q[3] * q[1] - q[0] * q[2])
    sinp = int(sinp)
    pitch = math.asin(sinp)
    return pitch


def get_bbox(center, size):
    minx = center[0] - (1 / 2) * size[0]
    maxx = center[0] + (1 / 2) * size[0]
    minz = center[2] - (1 / 2) * size[2]
    maxz = center[2] + (1 / 2) * size[2]
    return [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]


class CwahWall(Wall):
    def __init__(
        self,
        wall_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Geometry] = None,
        class_name: Optional[str] = None,
        room2room: Union[Tuple[str], str] = list(),
        empty: bool,
        **kwargs,
    ):
        super().__init__(
            wall_id,
            alias=alias,
            geometry=geometry,
            class_name=class_name,
            asset_id="not_exist",
            room2room=room2room,
            **kwargs,
        )
        self.empty = empty
        self.class_name = class_name

    @classmethod
    def create(cls, wall_data):
        polys_2d = Polygon2D(wall_data["polygon"])
        empty = wall_data.get("empty", False)
        return cls(
            wall_data["id"],
            geometry=polys_2d,
            class_name=wall_data["class_name"],
            props=wall_data,
            empty=empty,
        )

    def plot(self, axes=None):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.exterior.xy
        if self.empty:
            axes.plot(x, y, color="black", linestyle="-.", linewidth=0.5)
        else:
            axes.plot(x, y, color="black", linewidth=0.5)
        axes.fill(x, y, color="gray")

    def render(self, fig=None):
        if self.geometry is None:
            return
        if not fig:
            fig = go.Figure()
        x, y = self.geometry.shapely_geo.exterior.xy
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=self.geometry.x_min,
            y0=self.geometry.y_min,
            x1=self.geometry.x_max,
            y1=self.geometry.y_max,
            opacity=0.2,
            fillcolor="black",
            line=dict(width=0),
        )


class CwahDoor(Door):
    def __init__(
        self,
        door_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        class_name: Optional[str] = None,
        room2room: Tuple[str] = ...,
        openable: bool = True,
        is_open: bool = True,
        walls: Tuple[str] = ...,
        **kwargs,
    ):
        super().__init__(
            door_id,
            alias=alias,
            geometry=geometry,
            class_name=class_name,
            room2room=room2room,
            openable=openable,
            is_open=is_open,
            **kwargs,
        )
        self.walls = walls
        self.class_name = class_name

    @classmethod
    def create(cls, door):
        is_open = door.get("openness", 1) == 1
        openable = door.get("openable", False)
        polys_2d = Polygon2D(door["polygon"])
        room2room = [door["room0"], door["room1"]]
        class_name = door["class_name"]

        # "wall|3|10.14|3.38|15.21|3.38"

        return cls(
            door["id"],
            room2room=room2room,
            is_open=is_open,
            openable=openable,
            class_name=class_name,
            geometry=polys_2d,
        )

    def flip(self) -> None:
        """Flip doors wrt. wall attribute"""
        if len(self.walls) > 1 and "exterior" not in self.walls[1]:
            # Do not flip if the door is connected to outside.
            wall0, wall1 = self.walls
            self.walls = [wall1, wall0]
            self.room2room = [self.room2room[1], self.room2room[0]]

    def set_open(self, open=True):
        self.is_open = open

    def plot(self, axes=None):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.xy
        axes.plot(x, y, color="green", linewidth=3)

    def render(self, fig=None):
        if self.geometry is None:
            return
        if not fig:
            fig = go.Figure()
        x, y = self.geometry.shapely_geo.exterior.xy
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=self.geometry.x_min,
            y0=self.geometry.y_min,
            x1=self.geometry.x_max,
            y1=self.geometry.y_max,
            opacity=0.2,
            fillcolor="lightgreen",
            line=dict(width=0),
        )


class CwahWindow(Window):
    def __init__(
        self,
        window_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Geometry] = None,
        class_name: Optional[str] = None,
        room2room: Tuple[str] = ...,
        walls: Tuple[str] = ...,
        **kwargs,
    ):
        super().__init__(
            window_id,
            alias=alias,
            geometry=geometry,
            class_name=class_name,
            room2room=room2room,
            **kwargs,
        )
        self.walls = walls
        self.class_name = class_name

    @classmethod
    def create(cls, window):
        room2room = [window["room0"], window["room1"]]
        polys_2d = Polygon2D(window["polygon"])

        return cls(
            window["id"],
            geometry=polys_2d,
            room2room=room2room,
            class_name=window["class_name"],
        )

    def plot(self, axes=None):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.xy
        axes.plot(x, y, color="blue", linewidth=5)


class CwahRoom(Room):
    def __init__(
        self,
        obj_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        category: Optional[str] = None,
        class_name: Optional[str] = None,
        props: Dict[str, Any] = defaultdict(),
        **kwargs,
    ) -> None:
        super().__init__(
            obj_id,
            alias=alias,
            geometry=geometry,
            **kwargs,
        )
        self.class_name = class_name

    @classmethod
    def create(cls, room_data):
        polys_2d = None
        center = Point2D(
            room_data["bounding_box"]["center"][0],
            room_data["bounding_box"]["center"][2],
        )
        bounding_box = room_data["bounding_box"]
        size = bounding_box["size"]
        center = bounding_box["center"]

        bbox = get_bbox(center, size)
        rotation = ToEulerAngles(room_data["obj_transform"]["rotation"])
        polys_2d = None
        position = Point2D(
            room_data["obj_transform"]["position"][0],
            room_data["obj_transform"]["position"][2],
        )
        if bbox:
            # ul = center - Point2D(bbox["x"], bbox["z"]) * 0.5
            # br = center + Point2D(bbox["x"], bbox["z"]) * 0.5
            # polys_2d = Box2D(ul, br)
            polys_2d = Polygon2D(bbox)
            # TODO  Box2D rotate ISSUE
            # polys_2d.rotate(360 - rotation, origin=(center.x, center.y))

        return cls(
            room_data["id"],
            geometry=polys_2d,
            class_name=room_data["class_name"],
            props=room_data,
        )

    def plot(self, axes=None, color="aliceblue"):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.exterior.xy
        axes.fill(x, y, color=color)

    def render(self, fig=None):
        if self.geometry is None:
            return
        if not fig:
            fig = go.Figure()
        x, y = self.geometry.shapely_geo.exterior.xy
        fig.add_trace(
            go.Scatter(
                x=np.array(x),
                y=np.array(y),
                fill="toself",
                fillcolor="aliceblue",
                name=self.id,
                line=dict(color="gray"),
            )
        )


class CwahObject(Object2D):
    colorscales = list(CSS4_COLORS.keys())
    color_registry = defaultdict()

    def __init__(
        self,
        obj_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        category: Optional[str] = None,
        class_name: Optional[str] = None,
        props: Dict[str, Any] = defaultdict(),
        **kwargs,
    ) -> None:
        super().__init__(
            ObjectType.OBJECT,
            obj_id,
            alias=alias,
            geometry=geometry,
            **kwargs,
        )
        self.chilren_types = [ObjectType.OBJECT]
        if props is not None:
            self.props.update(props)

        self.category = category
        if self.category not in CwahObject.color_registry:
            select_color = random.choice(CwahObject.colorscales)
            CwahObject.color_registry.update({self.category: select_color})
            CwahObject.colorscales.remove(select_color)

        self.color = CwahObject.color_registry.get(self.category)
        self.position = self.geometry.centroid
        self.class_name = class_name

    @classmethod
    def create(cls, obj_data):
        if obj_data["category"] in ["Floor", "Ceiling", "Rooms", "Walls"]:
            obj_data["bounding_box"]["size"] = [0, 0, 0]
        category = obj_data["category"]
        class_name = obj_data["class_name"]
        object_id = obj_data["id"]

        props = obj_data
        bounding_box = obj_data["bounding_box"]
        size = bounding_box["size"]
        center = bounding_box["center"]

        bbox = get_bbox(center, size)
        rotation = ToEulerAngles(obj_data["obj_transform"]["rotation"])
        polys_2d = None
        position = Point2D(
            obj_data["obj_transform"]["position"][0],
            obj_data["obj_transform"]["position"][2],
        )

        if bbox:
            # ul = center - Point2D(bbox["x"], bbox["z"]) * 0.5
            # br = center + Point2D(bbox["x"], bbox["z"]) * 0.5
            # polys_2d = Box2D(ul, br)
            polys_2d = Polygon2D(bbox)
            # TODO  Box2D rotate ISSUE
            # polys_2d.rotate(360 - rotation, origin=(center.x, center.y))

        return cls(
            object_id,
            geometry=polys_2d,
            category=category,
            class_name=class_name,
            position=center,
            rotation=rotation,
            props=props,
        )

    def plot(self, axes=None):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.exterior.xy
        axes.fill(x, y)
        if len(self.children) > 0:
            for c in self.children:
                self.children[c].plot(axes=axes)

    def render(self, fig):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.exterior.xy
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=np.array(x),
                y=np.array(y),
                fill="toself",
                name=self.id,
                fillcolor=self.color,
                line=dict(width=0),
            )
        )
        if len(self.children) > 0:
            for c in self.children:
                self.children[c].render(fig)

    def __repr__(self) -> str:
        obj_string = f"class_name: {self.class_name}"
        return obj_string

    def get_obj_pose_info(self):
        if "openable" in self.props and self.props["openable"]:
            openness = self.props["openness"]
            openable = True
        else:
            openness = None
            openable = False
        if "pickupable" in self.props:
            pickupable = self.props["pickupable"]
        else:
            pickupable = False
        if "isBroken" in self.props:
            isBroken = self.props["isBroken"]
        else:
            isBroken = False
        return {
            "type": self.id.split("|")[0],
            "position": self.position,
            "rotation": self.rotation,
            "openable": openable,
            "openness": openness,
            "pickupable": pickupable,
            "broken": isBroken,
            "objectId": self.id,
            "name": self.id,
            "parentReceptacles": [],
            "bounding_box": None,
        }


@WORLD_REGISTRY.register()
class CwahWorld(World):
    EPSILON: float = 1e-3
    """Small value to compare floats within a bound."""

    def __init__(self, world_id: str) -> None:
        super().__init__(world_id)

    @classmethod
    def create(cls, world_config):
        world_id = world_config.get("id", "CwahWorld")
        world_data = copy.deepcopy(world_config["data"])

        world = cls(world_id)
        world.room_polygons = {}
        nodes = world_data["init_graph"]["nodes"]
        edges = world_data["init_graph"]["edges"]

        for node in nodes:
            if node["category"] == "Rooms":
                world.add_room(CwahRoom.create(node))

                bounding_box = node["bounding_box"]
                bbox = get_bbox(bounding_box["center"], bounding_box["size"])
                world.room_polygons[node["id"]] = bbox
                wall_idx = 1000
                for i in range(len(bbox)):
                    wall_data = {}
                    wall_polygon = [
                        bbox[i],
                        bbox[(i + 1) % len(bbox)],
                        bbox[(i + 1) % len(bbox)],
                        bbox[i],
                    ]
                    wall_data["polygon"] = wall_polygon
                    wall_data["id"] = wall_idx
                    wall_data["class_name"] = "Wall"
                    wall_idx += 1
                    world.add_wall(CwahWall.create(wall_data))
        for node in nodes:
            # if node['category'] == "Walls":
            #     world.add_wall(CwahWall.create(node))
            # the wall is not only the wall, but also huge obstacles
            if node["category"] == "Doors":
                door = {}
                door["id"] = node["id"]
                door["class_name"] = node["class_name"]
                bounding_box = node["bounding_box"]
                door["polygon"] = get_bbox(bounding_box["center"], bounding_box["size"])
                if "CAN_OPEN" in node["properties"]:
                    door["openable"] = True
                if "OPEN" in node["states"]:
                    door["openness"] = 1
                door["class_name"] = node["class_name"]
                rooms_between_door = []
                for e in edges:
                    if e["from_id"] == node["id"] and e["relation_type"] == "BETWEEN":
                        rooms_between_door.append(e["to_id"])
                if len(rooms_between_door) != 2:
                    door["room0"] = None
                    door["room1"] = None
                else:
                    door["room0"] = rooms_between_door[0]
                    door["room1"] = rooms_between_door[1]
                world.add_door(CwahDoor.create(door))
            elif node["category"] == "Windows":
                window = {}
                window["id"] = node["id"]
                window["class_name"] = node["class_name"]
                bounding_box = node["bounding_box"]

                window["polygon"] = get_bbox(
                    bounding_box["center"], bounding_box["size"]
                )
                if "CAN_OPEN" in node["properties"]:
                    window["openable"] = True
                if "OPEN" in node["states"]:
                    window["openness"] = 1
                window["class_name"] = node["class_name"]
                rooms_between_window = []
                for e in edges:
                    if e["from_id"] == node["id"] and e["relation_type"] == "BETWEEN":
                        rooms_between_window.append(e["to_id"])
                if len(rooms_between_window) != 2:
                    window["room0"] = None
                    window["room1"] = None
                else:
                    window["room0"] = rooms_between_window[0]
                    window["room1"] = rooms_between_window[1]

                world.add_window(CwahWindow.create(window))
            else:
                created_object = CwahObject.create(node)
                if created_object:
                    world.add_object(created_object)

        for node in nodes:
            world.id2object[node["id"]] = CwahObject.create(node)

        for e in edges:
            if e["relation_type"] == "ON":
                if e["from_id"] in world.rooms and e["to_id"] in world.rooms:
                    world.rooms[e["from_id"]].add_room(world.rooms[e["to_id"]])
                    world.rooms[e["to_id"]].add_room(world.rooms[e["from_id"]])
        logger.info(f"Successfully created world: {world_config['id']}")
        return world

    def add_room(self, room: CwahRoom) -> Optional[str]:
        self.rooms[room.id] = room
        return room.id

    def add_wall(self, wall: CwahWall) -> Optional[str]:
        logger.debug(wall.room2room)
        if len(wall.room2room) > 0:
            for r in wall.room2room:
                if r in self.rooms:
                    if self.rooms.get(r).add_wall(wall) is None:
                        return None

        self.walls[wall.id] = wall
        return wall.id

    def add_door(self, door: CwahDoor) -> Optional[str]:
        """
        Returns:
            door_id: return door_id if success, else return None
        """
        polys_2d = []
        # subwalls = []
        # wall0 = door.walls[0]
        # if wall0 not in self.walls:
        #     raise ValueError(f"Failed to add door {door.id}: No wall {wall0} found.")
        # wall0: Polygon2D = self.walls.get(wall0).geometry
        room2room = door.room2room
        # holePolygon = door.props['holePolygon']
        # start_door_poistion = holePolygon[0].x
        # door_width = holePolygon[1].x - start_door_poistion
        # door_open_size = min(door_width / 2.0, .5)
        # entrance_padding = 0.5
        # in_front_padding = 0.5

        # wall0 = [wall0.x_min, wall0.y_min, wall0.x_max, wall0.y_max]
        # if wall0.x_max - wall0.x_min < CwahWorld.EPSILON:
        #     x_wall = wall0.x_min

        #     # placed vertically
        #     flipped = wall0.coords[0].y > wall0.coords[1].y

        #     if flipped:
        #         polys_2d = Polygon2D([
        #             (x_wall - entrance_padding, wall0.y_max - start_door_poistion),
        #             (x_wall - entrance_padding, wall0.y_max - start_door_poistion - door_width),
        #             (x_wall + door_open_size + in_front_padding,
        #              wall0.y_max - start_door_poistion - door_width),
        #             (x_wall + door_open_size + in_front_padding, wall0.y_max - start_door_poistion)
        #         ])
        #         door.props.update(
        #             {"direction": "y_wall",
        #              "opened_geometry": [
        #                  Line2D([(x_wall, polys_2d.y_min), (polys_2d.x_max - in_front_padding, polys_2d.y_min)]),
        #                  Line2D([(x_wall, polys_2d.y_max), (polys_2d.x_max - in_front_padding, polys_2d.y_max)]),
        #              ],
        #              "closed_geometry": Line2D([(x_wall, polys_2d.y_min), (x_wall, polys_2d.y_max)])
        #              })
        #     else:
        #         polys_2d = Polygon2D([
        #             (x_wall - door_open_size - in_front_padding, wall0.y_min + start_door_poistion),
        #             (x_wall - door_open_size - in_front_padding,
        #              wall0.y_min + start_door_poistion + door_width),
        #             (x_wall + entrance_padding, wall0.y_min + start_door_poistion + door_width),
        #             (x_wall + entrance_padding, wall0.y_min + start_door_poistion)
        #         ])
        #         door.props.update(
        #             {"direction": "y_wall",
        #              "opened_geometry": [
        #                  Line2D([(x_wall, polys_2d.y_min), (polys_2d.x_min + in_front_padding, polys_2d.y_min)]),
        #                  Line2D([(x_wall, polys_2d.y_max), (polys_2d.x_min + in_front_padding, polys_2d.y_max)]),
        #              ],
        #              "closed_geometry": Line2D([(x_wall, polys_2d.y_min), (x_wall, polys_2d.y_max)])
        #              })
        # else:
        #     z_wall = wall0.y_min

        #     # placed along x
        #     flipped = wall0.coords[0].x > wall0.coords[1].x
        #     if flipped:
        #         polys_2d = Polygon2D([
        #             (wall0.x_max - start_door_poistion, z_wall - door_open_size - in_front_padding),
        #             (wall0.x_max - start_door_poistion - door_width,
        #              z_wall - door_open_size - in_front_padding),
        #             (wall0.x_max - start_door_poistion - door_width, z_wall + entrance_padding),
        #             (wall0.x_max - start_door_poistion, z_wall + entrance_padding)
        #         ])
        #         door.props.update(
        #             {"direction": "x_wall",
        #              "opened_geometry": [
        #                  Line2D([(polys_2d.x_min, z_wall), (polys_2d.x_min, polys_2d.y_min + in_front_padding)]),
        #                  Line2D([(polys_2d.x_max, z_wall), (polys_2d.x_max, polys_2d.y_min + in_front_padding)]),
        #              ],
        #              "closed_geometry": Line2D([(polys_2d.x_min, z_wall), (polys_2d.x_max, z_wall)])
        #              })
        #     else:
        #         polys_2d = Polygon2D([
        #             (wall0.x_min + start_door_poistion, z_wall - entrance_padding),
        #             (wall0.x_min + start_door_poistion + door_width, z_wall - entrance_padding),
        #             (wall0.x_min + start_door_poistion + door_width,
        #              z_wall + door_open_size + in_front_padding),
        #             (wall0.x_min + start_door_poistion, z_wall + door_open_size + in_front_padding)
        #         ])
        #         door.props.update(
        #             {"direction": "x_wall",
        #              "opened_geometry": [
        #                  Line2D([(polys_2d.x_min, z_wall), (polys_2d.x_min, polys_2d.y_max - in_front_padding)]),
        #                  Line2D([(polys_2d.x_max, z_wall), (polys_2d.x_max, polys_2d.y_max - in_front_padding)]),
        #              ],
        #              "closed_geometry": Line2D([(polys_2d.x_min, z_wall), (polys_2d.x_max, z_wall)])
        #              })

        # door.geometry = polys_2d
        # door.props.update(dict(
        #     start_door_poistion=start_door_poistion,
        #     door_width=door_width,
        #     door_open_size=door_open_size,
        #     entrance_padding=entrance_padding,
        #     in_front_padding=in_front_padding
        # ))
        # for wall in door.walls:
        #     self.walls.get(wall).add_door(door)
        for room in room2room:
            if room:
                self.rooms.get(room).add_door(door)
        self.doors[door.id] = door

    def add_window(self, window: CwahWindow):
        for room in window.room2room:
            if room in self.rooms:
                self.rooms[room].add_window(window)
        # for wall in window.walls:
        #     if wall in self.walls:
        #         self.walls[wall].add_window(window)
        self.windows[window.id] = window
        return window.id

    def add_object(self, object: CwahObject):
        self.objects[object.id] = object
        return object.id

    def contains_object(self, object_id: str) -> bool:
        return object_id in self.objects

    def get_object(self, object_id: str):
        if not self.contains_object(object_id):
            return None
        return self.objects[object_id]

    def pop_object(self, object_id: str):
        if not self.contains_object(object_id):
            return None

        obj = self.objects.get(object_id)

        del self.objects[object_id]
        return obj
