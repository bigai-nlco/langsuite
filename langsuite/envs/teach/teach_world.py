# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from langsuite.constants import CSS4_COLORS
from langsuite.shapes import Geometry, Point2D, Polygon2D
from langsuite.utils.logging import logger
from langsuite.world import WORLD_REGISTRY, Object2D, ObjectType, Room, Wall, World

TeachPath = Path(__file__).parent


class TeachWall(Wall):
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


class TeachRoom(Room):
    @classmethod
    def create(cls, room_data):
        polys_3d = room_data["floorPolygon"]
        polys_2d = []
        for p in polys_3d:
            polys_2d.append((p[0], p[1]))
        polys_2d = Polygon2D(polys_2d)
        return cls(room_data["id"], geometry=polys_2d, asset_id=room_data["roomType"])

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


class TeachObject(Object2D):
    colorscales = list(CSS4_COLORS.keys())
    color_registry = defaultdict()

    def __init__(
        self,
        obj_id: str,
        *,
        alias: Optional[str] = None,
        geometry: Optional[Polygon2D] = None,
        asset_id: Optional[str] = None,
        position: Point2D = None,
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
        if self.category not in TeachObject.color_registry:
            select_color = random.choice(TeachObject.colorscales)
            TeachObject.color_registry.update({self.category: select_color})
            TeachObject.colorscales.remove(select_color)

        self.color = TeachObject.color_registry.get(self.category)
        # self.position = self.geometry.centroid
        self.position = position
        self.rotation = rotation

    @classmethod
    def create(cls, obj_data, objs_data):
        if "Floor" in obj_data["objectId"]:
            obj_data["axisAlignedBoundingBox"]["size"] = {"x": 0, "y": 0, "z": 0}
        asset_id = obj_data["objectType"]
        object_id = obj_data["objectId"]

        props = obj_data

        size = obj_data.get("axisAlignedBoundingBox").get("size")
        center = obj_data.get("axisAlignedBoundingBox").get("center")

        def get_bbox(center, size):
            minx = center["x"] - (1 / 2) * size["x"]
            maxx = center["x"] + (1 / 2) * size["x"]
            minz = center["z"] - (1 / 2) * size["z"]
            maxz = center["z"] + (1 / 2) * size["z"]
            return [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]

        # bbox = get_bbox(bbox_cornerpoints)
        bbox = get_bbox(center, size)
        rotation = obj_data["rotation"]["y"]
        polys_2d = None
        center = Point2D(center["x"], center["z"])
        position = Point2D(
            obj_data.get("position").get("x"), obj_data.get("position").get("z")
        )

        if bbox:
            # ul = center - Point2D(bbox["x"], bbox["z"]) * 0.5
            # br = center + Point2D(bbox["x"], bbox["z"]) * 0.5
            # polys_2d = Box2D(ul, br)
            polys_2d = Polygon2D(bbox)
            # TODO  Box2D rotate ISSUE
            # polys_2d.rotate(360 - rotation, origin=(center.x, center.y))

        children = {}
        if (
            "receptacleObjectIds" in obj_data
            and obj_data["receptacleObjectIds"] is not None
            and len(obj_data["receptacleObjectIds"]) > 0
        ):
            children_id = deepcopy(obj_data["receptacleObjectIds"])

            while len(children_id) > 0:
                c_id = children_id.pop(0)
                for object_c in objs_data:
                    if object_c["objectId"] == c_id:
                        children[c_id] = TeachObject.create(object_c, objs_data)
                        break

        return cls(
            object_id,
            geometry=polys_2d,
            asset_id=asset_id,
            position=center,
            rotation=rotation,
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
                if child.del_children(child_id):
                    return True

        return False

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
class TeachWorld(World):
    EPSILON: float = 1e-3
    """Small value to compare floats within a bound."""

    @classmethod
    def create(cls, world_config):
        world_id = world_config.get("id", "TeachWorld")
        world = cls(world_id)

        floor_plan2polygons_path = world_config["floor_plan_path"]
        with open(floor_plan2polygons_path) as f:
            floor_plan2polygons = json.load(f)

        world_data = world_config["data"]
        room = {}
        room["id"] = world_data["tasks"][0]["episodes"][0]["world"].split("_")[0]
        room["floorPolygon"] = floor_plan2polygons[room["id"]]

        kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
        living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
        bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
        bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

        if room["id"] in kitchens:
            room["roomType"] = "Kitchen"
        elif room["id"] in living_rooms:
            room["roomType"] = "Living Room"
        elif room["id"] in bedrooms:
            room["roomType"] = "Bedroom"
        elif room["id"] in bathrooms:
            room["roomType"] = "Bathroom"

        world.add_room(TeachRoom.create(room))
        room_polygons = room["floorPolygon"]
        for i in range(len(room_polygons)):
            wall_polygon = [
                room_polygons[i],
                room_polygons[(i + 1) % len(room_polygons)],
                room_polygons[(i + 1) % len(room_polygons)],
                room_polygons[i],
            ]
            world.add_wall(TeachWall.create("wall" + str(i), wall_polygon))
        init_state = world_data["tasks"][0]["episodes"][0]["initial_state"]

        for obj in init_state["objects"]:
            parentReceptacles = obj["parentReceptacles"]
            parent = []
            if parentReceptacles is not None:
                for r in parentReceptacles:
                    if "Floor" in r:
                        continue
                    else:
                        parent.append(r)
            if "Floor" not in obj["objectId"] and len(parent) == 0:
                created_object = TeachObject.create(obj, init_state["objects"])
                if created_object:
                    world.add_object(created_object)

        floor_plan_controlled_objects_path = world_config[
            "floor_plan_controlled_objects_path"
        ]
        with open(floor_plan_controlled_objects_path) as f:
            floor_plan_controlled_objects = json.load(f)
        world.controlled_objects = floor_plan_controlled_objects[room["id"]]
        world.room_polygons = room_polygons

        logger.info(f"Successfully created world: {world_config['id']}")
        for obj_id, obj in world.objects.items():
            world.id2object[obj_id] = obj
            children = obj.find_all_children()
            for child in children:
                world.id2object[child.id] = child
        for obj_id, obj in init_state["custom_object_metadata"].items():
            if "Floor" in obj_id:
                continue
            if obj_id not in world.id2object:
                continue
            world.id2object[obj_id].props.update(obj)
        return world

    def add_room(self, room: TeachRoom) -> Optional[str]:
        self.rooms[room.id] = room
        return room.id

    def add_wall(self, wall: TeachWall) -> Optional[str]:
        logger.debug(wall.room2room)
        if len(wall.room2room) > 0:
            for r in wall.room2room:
                if r in self.rooms:
                    if self.rooms.get(r).add_wall(wall) is None:
                        return None

        self.walls[wall.id] = wall
        return wall.id

    def add_object(self, object: TeachObject):
        self.objects[object.id] = object
        return object.id

    def contains_object(self, object_id: str) -> bool:
        return object_id in self.id2object

    def get_object(self, object_id: str):
        if not self.contains_object(object_id):
            return None
        return self.id2object[object_id]

    def pop_object(self, object_id: str):
        if not self.contains_object(object_id):
            return False

        if object_id not in self.objects:
            obj = self.id2object[object_id]
            for child in obj.children.values():
                if child.del_child(object_id):
                    return True
        else:
            del self.objects[object_id]
            return True
        return False
