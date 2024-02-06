# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations


# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from langsuite.constants import CSS4_COLORS
from langsuite.shapes import Geometry, Point2D, Polygon2D
from langsuite.utils.logging import logger
from langsuite.world import WORLD_REGISTRY, Object2D, ObjectType, Room, Wall, World

AlfredPath = Path(__file__).parent


class AlfredWall(Wall):
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
    def create(cls, id, polys_2d):
        empty = False
        polys_2d = Polygon2D(polys_2d)

        return cls(id, geometry=polys_2d, empty=empty)

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


class AlfredRoom(Room):
    @classmethod
    def create(cls, room_data):
        polys_3d = room_data["floorPolygon"]
        polys_2d = []
        for p in polys_3d:
            polys_2d.append((p[0], p[1]))
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


class AlfredObject(Object2D):
    colorscales = list(CSS4_COLORS.keys())
    color_registry = defaultdict()

    def __init__(
        self,
        obj_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        asset_id: Optional[str] = None,
        rotation: float = 0,
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
        if self.category not in AlfredObject.color_registry:
            select_color = random.choice(AlfredObject.colorscales)
            AlfredObject.color_registry.update({self.category: select_color})
            AlfredObject.colorscales.remove(select_color)

        self.color = AlfredObject.color_registry.get(self.category)
        self.position = self.geometry.centroid
        self.rotation = rotation

    @classmethod
    def create(cls, obj_data, objs_data):
        if "bbox" not in obj_data.keys():
            # TODO data not provided, use a small size to simplify the problem
            size = {"x": 0.1, "z": 0.1}
        else:
            size = {"x": obj_data["bbox"]["x"], "z": obj_data["bbox"]["z"]}
        rotation = obj_data["rotation"]["y"]
        polys_2d = None
        if "center" in obj_data.keys():
            center = Point2D(obj_data["center"]["x"], obj_data["center"]["z"])
        else:
            center = Point2D(obj_data["position"]["x"], obj_data["position"]["z"])
        minx = center.x - size["x"] / 2
        maxx = center.x + size["x"] / 2
        minz = center.y - size["z"] / 2
        maxz = center.y + size["z"] / 2
        bbox = [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]
        if bbox:
            # ul = center - Point2D(bbox["x"], bbox["z"]) * 0.5
            # br = center + Point2D(bbox["x"], bbox["z"]) * 0.5
            # polys_2d = Box2D(ul, br)
            polys_2d = Polygon2D(bbox)
            # TODO  Box2D rotate ISSUE
            # polys_2d.rotate(360 - rotation, origin=(center.x, center.y))

        children = defaultdict()

        if (
            "receptacleObjectIds" in obj_data
            and obj_data["receptacleObjectIds"] is not None
            and len(obj_data["receptacleObjectIds"]) > 0
        ):
            children_id = obj_data["receptacleObjectIds"]
            for c_id in children_id:
                for object_c in objs_data:
                    if object_c["objectId"] == c_id:
                        child = AlfredObject.create(object_c, objs_data)
                        children[c_id] = child
                        break

        return cls(
            obj_data["objectId"],
            geometry=polys_2d,
            asset_id="alfred_asset_" + obj_data["objectId"],
            rotation=rotation,
            children=children,
            props=obj_data,
        )

    def update(self, position):
        diff_x = position.x - self.position.x
        diff_y = position.y - self.position.y
        self.position = position
        x, y = self.geometry.shapely_geo.exterior.xy
        bbox = []
        for x_i, y_i in zip(x, y):
            bbox.append([x_i + diff_x, y_i + diff_y])
            if len(bbox) == 4:
                break
        polys_2d = Polygon2D(bbox)
        self.geometry = polys_2d

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

    def find_all_children(self):
        children = []
        if len(self.children) > 0:
            for child in self.children.values():
                children.append(child)
                children.extend(child.find_all_children())
        return children

    def del_child(self, child_id):
        if child_id in self.children:
            del self.children[child_id]
            return True
        else:
            for child in self.children.values():
                if child.del_child(child_id):
                    return True

        return False

    def get_child(self, child_id):
        if child_id in self.children:
            return self.children[child_id]
        else:
            for child in self.children.values():
                child_child = child.get_child(child_id)
                if child_child is not None:
                    return child_child

        return None


@WORLD_REGISTRY.register()
class AlfredWorld(World):
    EPSILON: float = 1e-3
    """Small value to compare floats within a bound."""

    @classmethod
    def create(cls, world_config):
        world_id = world_config.get("id", "AlfredWorld")
        world = cls(world_id)

        # asset_path = world_config["asset_path"]
        # metadata_path = world_config["metadata_path"]
        # receptacles_path = world_config["receptacles_path"]

        # assets = io_utils.load_assets(asset_path)
        # metadata = io_utils.load_object_metadata(metadata_path)
        # receptacles = io_utils.load_receptacles(receptacles_path)

        world_data = world_config["data"]
        room = {}
        room["floorPolygon"] = world_data["polys_2d"]
        room["id"] = world_data["floor_plan"]
        room_polygons = world_data["polys_2d"]
        for i in range(len(room_polygons)):
            wall_polygon = [
                room_polygons[i],
                room_polygons[(i + 1) % len(room_polygons)],
                room_polygons[(i + 1) % len(room_polygons)],
                room_polygons[i],
            ]
            world.add_wall(AlfredWall.create("wall" + str(i), wall_polygon))
        world.add_room(AlfredRoom.create(room))
        for object in world_data["objects"]:
            parentReceptacles = object["parentReceptacles"]
            parent = []
            if parentReceptacles is not None:
                for r in parentReceptacles:
                    if "Floor" in r:
                        continue
                    else:
                        parent.append(r)
            if (
                "FloorLamp" in object["objectId"] or "Floor" not in object["objectId"]
            ) and len(parent) == 0:
                world.add_object(AlfredObject.create(object, world_data["objects"]))
        world.room_polygons = room_polygons
        logger.info(f"Successfully created world: {world_config['id']}")
        for obj_id, obj in world.objects.items():
            world.id2object[obj_id] = obj
            children = obj.find_all_children()
            for child in children:
                world.id2object[child.id] = child
        logger.info(f"Successfully created world: {world_config['id']}")
        return world

    def add_room(self, room: AlfredRoom) -> Optional[str]:
        self.rooms[room.id] = room
        return room.id

    def add_wall(self, wall: AlfredWall) -> Optional[str]:
        logger.debug(wall.room2room)
        if len(wall.room2room) > 0:
            for r in wall.room2room:
                if r in self.rooms:
                    if self.rooms.get(r).add_wall(wall) is None:
                        return None

        self.walls[wall.id] = wall
        return wall.id

    def add_object(self, object: AlfredObject):
        self.objects[object.id] = object
        return object.id

    def contains_object(self, object_id: str) -> bool:
        obj = self.get_object(object_id)
        if obj is not None:
            return True
        else:
            return False

    def get_object(self, object_id: str):
        if object_id not in self.objects:
            for obj in self.objects.values():
                child = obj.get_child(object_id)
                if child is not None:
                    return child
        else:
            return self.objects[object_id]

        return None

    def pop_object(self, object_id: str):
        if not self.contains_object(object_id):
            return False

        if object_id not in self.objects:
            for obj in self.objects.values():
                if obj.del_child(object_id):
                    return True
        else:
            del self.objects[object_id]
            return True
        return False
