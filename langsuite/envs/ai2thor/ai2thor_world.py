# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from langsuite.constants import CSS4_COLORS
from langsuite.envs.ai2thor import ai2thor_utils
from langsuite.shapes import Box2D, Geometry, Line2D, Point2D, Polygon2D
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

AI2THORPath = Path(__file__).parent


class AI2THORWall(Wall):
    def __init__(
        self,
        wall_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Geometry] = None,
        asset_id: Optional[str] = None,
        room2room: Union[Tuple[str], str] = list(),
        empty: bool,
        **kwargs,
    ):
        super().__init__(
            wall_id,
            alias=alias,
            geometry=geometry,
            asset_id=asset_id,
            room2room=room2room,
            **kwargs,
        )
        self.empty = empty

    @classmethod
    def create(cls, wall):
        polys_3d = wall["polygon"]
        polys_2d = []
        empty = wall.get("empty", False)

        for dot in polys_3d:
            polys_2d.append((dot["x"], dot["z"]))
        logger.debug(polys_2d)
        polys_2d = Polygon2D(polys_2d)

        return cls(wall["id"], geometry=polys_2d, empty=empty)

    def plot(self, axes=None):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.exterior.xy
        if self.empty:
            axes.plot(x, y, color="black", linestyle="-.", linewidth=0.5)
        else:
            axes.plot(x, y, color="black", linewidth=0.5)
        axes.fill(x, y, color="gray")

    def render(self):
        if not self.geometry:
            return


class AI2THORDoor(Door):
    def __init__(
        self,
        door_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        asset_id: Optional[str] = None,
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
            asset_id=asset_id,
            room2room=room2room,
            openable=openable,
            is_open=is_open,
            **kwargs,
        )
        self.walls = walls

    @classmethod
    def create(cls, door):
        is_open = door.get("openness", 1) == 1
        openable = door.get("openable", False)
        holePolygon = [Point2D(p["x"], p["y"]) for p in door["holePolygon"]]
        room2room = [door["room0"], door["room1"]]
        asset_id = door["assetId"]

        # "wall|3|10.14|3.38|15.21|3.38"

        return cls(
            door["id"],
            room2room=room2room,
            is_open=is_open,
            openable=openable,
            asset_id=asset_id,
            walls=[door["wall0"], door["wall1"]],
            holePolygon=holePolygon,
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

        if self.is_open:
            door_x = (
                [c.x for c in self.props["opened_geometry"][0].coords]
                + [None]
                + [c.x for c in self.props["opened_geometry"][1].coords]
            )
            door_y = (
                [c.y for c in self.props["opened_geometry"][0].coords]
                + [None]
                + [c.y for c in self.props["opened_geometry"][1].coords]
            )
        else:
            door_x = [c.x for c in self.props["closed_geometry"].coords]
            door_y = [c.y for c in self.props["closed_geometry"].coords]
        fig.add_trace(
            go.Scatter(
                x=door_x,
                y=door_y,
                line=dict(color="green"),
                name=f"{self.id}|{'opened' if self.is_open else 'closed'}",
            )
        )


class AI2THORWindow(Window):
    def __init__(
        self,
        window_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Geometry] = None,
        asset_id: Optional[str] = None,
        room2room: Tuple[str] = ...,
        walls: Tuple[str] = ...,
        **kwargs,
    ):
        super().__init__(
            window_id,
            alias=alias,
            geometry=geometry,
            asset_id=asset_id,
            room2room=room2room,
            **kwargs,
        )
        self.walls = walls

    @classmethod
    def create(cls, window):
        polygon_dots = window["holePolygon"]
        room2room = [window["room0"], window["room1"]]
        wall0 = window["wall0"].split("|")[2:]
        wall0 = [float(w) for w in wall0]
        polys_2d = []

        if wall0[0] == wall0[2]:
            polys_2d.append((wall0[0], wall0[1] + polygon_dots[0]["x"]))
            polys_2d.append((wall0[0], wall0[1] + polygon_dots[1]["x"]))
        elif wall0[1] == wall0[3]:
            polys_2d.append((wall0[2] - polygon_dots[1]["x"], wall0[1]))
            polys_2d.append((wall0[2] - polygon_dots[0]["x"], wall0[1]))
        polys_2d = Line2D(polys_2d)
        return cls(
            window["id"],
            geometry=polys_2d,
            room2room=room2room,
            walls=[window["wall0"], window["wall1"]],
            asset_id=window["assetId"],
        )

    def plot(self, axes=None):
        if self.geometry is None:
            return
        x, y = self.geometry.shapely_geo.xy
        axes.plot(x, y, color="blue", linewidth=5)


class AI2THORRoom(Room):
    @classmethod
    def create(cls, room_data):
        polys_3d = room_data["floorPolygon"]
        polys_2d = []
        for p in polys_3d:
            polys_2d.append((p["x"], p["z"]))
        polys_2d = Polygon2D(polys_2d)
        return cls(room_data["id"], geometry=polys_2d)

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


class AI2THORObject(Object2D):
    colorscales = list(CSS4_COLORS.keys())
    color_registry = defaultdict()

    def __init__(
        self,
        obj_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        asset_id: Optional[str] = None,
        props: Dict[str, Any] = defaultdict(),
        **kwargs,
    ) -> None:
        super().__init__(
            ObjectType.OBJECT,
            obj_id,
            alias=alias,
            geometry=geometry,
            asset_id=asset_id,
            **kwargs,
        )
        self.chilren_types = [ObjectType.OBJECT]
        if props is not None:
            self.props.update(props)

        self.category = self.id.split("|")[0]
        if self.category not in AI2THORObject.color_registry:
            select_color = random.choice(AI2THORObject.colorscales)
            AI2THORObject.color_registry.update({self.category: select_color})
            AI2THORObject.colorscales.remove(select_color)

        self.color = AI2THORObject.color_registry.get(self.category)
        self.position = self.geometry.centroid

    @classmethod
    def create(cls, obj_data, obj_props):
        asset_id = obj_data["assetId"]
        if asset_id in obj_props:
            props = obj_props[asset_id].copy()
        else:
            props = dict()

        bbox = props.get("bbox")
        rotation = obj_data["rotation"]["y"]
        polys_2d = None
        center = Point2D(obj_data["position"]["x"], obj_data["position"]["z"])
        if bbox:
            ul = center - Point2D(bbox["x"], bbox["z"]) * 0.5
            br = center + Point2D(bbox["x"], bbox["z"]) * 0.5
            polys_2d = Box2D(ul, br)
            polys_2d.rotate(360 - rotation, origin=(center.x, center.y))

        children = []
        if "children" in obj_data:
            for child in obj_data["children"]:
                children.append(AI2THORObject.create(child, obj_props))
        return cls(
            obj_data["id"],
            geometry=polys_2d,
            asset_id=asset_id,
            children=children,
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
                c.render(fig)


@WORLD_REGISTRY.register()
class AI2THORWorld(World):
    EPSILON: float = 1e-3
    """Small value to compare floats within a bound."""

    @classmethod
    def create(cls, world_config):
        world_id = world_config.get("id", "AI2THORWorld")
        world = cls(world_id)

        asset_path = world_config["asset_path"]
        # metadata_path = world_config["metadata_path"]
        # receptacles_path = world_config["receptacles_path"]

        assets = ai2thor_utils.load_assets(asset_path)
        # metadata = ai2thor_utils.load_object_metadata(metadata_path)
        # receptacles = ai2thor_utils.load_receptacles(receptacles_path)

        world_data = world_config["data"]
        for room in world_data["rooms"]:
            world.add_room(AI2THORRoom.create(room))
        for wall in world_data["walls"]:
            world.add_wall(AI2THORWall.create(wall))
        for door in world_data["doors"]:
            world.add_door(AI2THORDoor.create(door))
        for window in world_data["windows"]:
            world.add_window(AI2THORWindow.create(window))
        for object in world_data["objects"]:
            world.add_object(AI2THORObject.create(object, assets))

        logger.info(f"Successfully created world: {world_config['id']}")
        return world

    def add_room(self, room: AI2THORRoom) -> Optional[str]:
        self.rooms[room.id] = room
        return room.id

    def add_wall(self, wall: AI2THORWall) -> Optional[str]:
        logger.debug(wall.room2room)
        if len(wall.room2room) > 0:
            for r in wall.room2room:
                if r in self.rooms:
                    if self.rooms.get(r).add_wall(wall) is None:
                        return None

        self.walls[wall.id] = wall
        return wall.id

    def add_door(self, door: AI2THORDoor) -> Optional[str]:
        """
        Returns:
            door_id: return door_id if success, else return None
        """
        polys_2d = []
        # subwalls = []
        wall0 = door.walls[0]
        if wall0 not in self.walls:
            raise ValueError(f"Failed to add door {door.id}: No wall {wall0} found.")
        wall0: Polygon2D = self.walls.get(wall0).geometry
        room2room = door.room2room
        holePolygon = door.props["holePolygon"]
        start_door_poistion = holePolygon[0].x
        door_width = holePolygon[1].x - start_door_poistion
        door_open_size = min(door_width / 2.0, 0.5)
        entrance_padding = 0.5
        in_front_padding = 0.5

        # wall0 = [wall0.x_min, wall0.y_min, wall0.x_max, wall0.y_max]
        if wall0.x_max - wall0.x_min < AI2THORWorld.EPSILON:
            x_wall = wall0.x_min

            # placed vertically
            flipped = wall0.coords[0].y > wall0.coords[1].y

            if flipped:
                polys_2d = Polygon2D(
                    [
                        (x_wall - entrance_padding, wall0.y_max - start_door_poistion),
                        (
                            x_wall - entrance_padding,
                            wall0.y_max - start_door_poistion - door_width,
                        ),
                        (
                            x_wall + door_open_size + in_front_padding,
                            wall0.y_max - start_door_poistion - door_width,
                        ),
                        (
                            x_wall + door_open_size + in_front_padding,
                            wall0.y_max - start_door_poistion,
                        ),
                    ]
                )
                door.props.update(
                    {
                        "direction": "y_wall",
                        "opened_geometry": [
                            Line2D(
                                [
                                    (x_wall, polys_2d.y_min),
                                    (polys_2d.x_max - in_front_padding, polys_2d.y_min),
                                ]
                            ),
                            Line2D(
                                [
                                    (x_wall, polys_2d.y_max),
                                    (polys_2d.x_max - in_front_padding, polys_2d.y_max),
                                ]
                            ),
                        ],
                        "closed_geometry": Line2D(
                            [(x_wall, polys_2d.y_min), (x_wall, polys_2d.y_max)]
                        ),
                    }
                )
            else:
                polys_2d = Polygon2D(
                    [
                        (
                            x_wall - door_open_size - in_front_padding,
                            wall0.y_min + start_door_poistion,
                        ),
                        (
                            x_wall - door_open_size - in_front_padding,
                            wall0.y_min + start_door_poistion + door_width,
                        ),
                        (
                            x_wall + entrance_padding,
                            wall0.y_min + start_door_poistion + door_width,
                        ),
                        (x_wall + entrance_padding, wall0.y_min + start_door_poistion),
                    ]
                )
                door.props.update(
                    {
                        "direction": "y_wall",
                        "opened_geometry": [
                            Line2D(
                                [
                                    (x_wall, polys_2d.y_min),
                                    (polys_2d.x_min + in_front_padding, polys_2d.y_min),
                                ]
                            ),
                            Line2D(
                                [
                                    (x_wall, polys_2d.y_max),
                                    (polys_2d.x_min + in_front_padding, polys_2d.y_max),
                                ]
                            ),
                        ],
                        "closed_geometry": Line2D(
                            [(x_wall, polys_2d.y_min), (x_wall, polys_2d.y_max)]
                        ),
                    }
                )
        else:
            z_wall = wall0.y_min

            # placed along x
            flipped = wall0.coords[0].x > wall0.coords[1].x
            if flipped:
                polys_2d = Polygon2D(
                    [
                        (
                            wall0.x_max - start_door_poistion,
                            z_wall - door_open_size - in_front_padding,
                        ),
                        (
                            wall0.x_max - start_door_poistion - door_width,
                            z_wall - door_open_size - in_front_padding,
                        ),
                        (
                            wall0.x_max - start_door_poistion - door_width,
                            z_wall + entrance_padding,
                        ),
                        (wall0.x_max - start_door_poistion, z_wall + entrance_padding),
                    ]
                )
                door.props.update(
                    {
                        "direction": "x_wall",
                        "opened_geometry": [
                            Line2D(
                                [
                                    (polys_2d.x_min, z_wall),
                                    (polys_2d.x_min, polys_2d.y_min + in_front_padding),
                                ]
                            ),
                            Line2D(
                                [
                                    (polys_2d.x_max, z_wall),
                                    (polys_2d.x_max, polys_2d.y_min + in_front_padding),
                                ]
                            ),
                        ],
                        "closed_geometry": Line2D(
                            [(polys_2d.x_min, z_wall), (polys_2d.x_max, z_wall)]
                        ),
                    }
                )
            else:
                polys_2d = Polygon2D(
                    [
                        (wall0.x_min + start_door_poistion, z_wall - entrance_padding),
                        (
                            wall0.x_min + start_door_poistion + door_width,
                            z_wall - entrance_padding,
                        ),
                        (
                            wall0.x_min + start_door_poistion + door_width,
                            z_wall + door_open_size + in_front_padding,
                        ),
                        (
                            wall0.x_min + start_door_poistion,
                            z_wall + door_open_size + in_front_padding,
                        ),
                    ]
                )
                door.props.update(
                    {
                        "direction": "x_wall",
                        "opened_geometry": [
                            Line2D(
                                [
                                    (polys_2d.x_min, z_wall),
                                    (polys_2d.x_min, polys_2d.y_max - in_front_padding),
                                ]
                            ),
                            Line2D(
                                [
                                    (polys_2d.x_max, z_wall),
                                    (polys_2d.x_max, polys_2d.y_max - in_front_padding),
                                ]
                            ),
                        ],
                        "closed_geometry": Line2D(
                            [(polys_2d.x_min, z_wall), (polys_2d.x_max, z_wall)]
                        ),
                    }
                )

        door.geometry = polys_2d
        door.props.update(
            dict(
                start_door_poistion=start_door_poistion,
                door_width=door_width,
                door_open_size=door_open_size,
                entrance_padding=entrance_padding,
                in_front_padding=in_front_padding,
            )
        )
        for wall in door.walls:
            self.walls.get(wall).add_door(door)
        for room in room2room:
            self.rooms.get(room).add_door(door)
        self.doors[door.id] = door

    def add_window(self, window: AI2THORWindow):
        for room in window.room2room:
            if room in self.rooms:
                self.rooms[room].add_window(window)
        for wall in window.walls:
            if wall in self.walls:
                self.walls[wall].add_window(window)
        self.windows[window.id] = window
        return window.id

    def add_object(self, object: AI2THORObject):
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
