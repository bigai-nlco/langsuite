# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from langsuite.constants import CSS4_COLORS
from langsuite.shapes import Geometry, Point2D, Polygon2D
from langsuite.utils.logging import logger
from langsuite.world import WORLD_REGISTRY, Object2D, ObjectType, Room, Wall, World


class RearrangeWall(Wall):
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


class RearrangeRoom(Room):
    @classmethod
    def create(cls, id, polys_2d):
        polys_2d = Polygon2D(polys_2d)
        return cls(id, geometry=polys_2d)

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


class RearrangeObject(Object2D):
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
        if self.category not in RearrangeObject.color_registry:
            select_color = random.choice(RearrangeObject.colorscales)
            RearrangeObject.color_registry.update({self.category: select_color})
            RearrangeObject.colorscales.remove(select_color)

        self.color = RearrangeObject.color_registry.get(self.category)
        self.position = self.geometry.centroid
        self.rotation = rotation

    @classmethod
    def create(cls, obj_data, objs_data):
        asset_id = obj_data["name"]
        object_id = obj_data["objectId"]

        props = obj_data

        # bbox_cornerpoints = obj_data.get("axisAlignedBoundingBox").get("cornerPoints")
        size = obj_data.get("axisAlignedBoundingBox").get("size")
        # position = obj_data.get('position')
        position = obj_data.get("axisAlignedBoundingBox").get("center")

        # def get_bbox(cornerPoints):
        #     minx = min([x[0] for x in cornerPoints])
        #     maxx = max([x[0] for x in cornerPoints])
        #     minz = min([x[2] for x in cornerPoints])
        #     maxz = max([x[2] for x in cornerPoints])
        #     return [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]

        def get_bbox(center, size):
            size_x = size["x"]
            size_z = size["z"]
            if size_x == 0:
                size_x = 0.1
            if size_z == 0:
                size_z = 0.1
            minx = center["x"] - (1 / 2) * size_x
            maxx = center["x"] + (1 / 2) * size_x
            minz = center["z"] - (1 / 2) * size_z
            maxz = center["z"] + (1 / 2) * size_z
            return [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]

        # bbox = get_bbox(bbox_cornerpoints)
        bbox = get_bbox(position, size)
        rotation = obj_data["rotation"]["y"]
        polys_2d = None
        center = Point2D(position["x"], position["z"])
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
            children_id = obj_data["receptacleObjectIds"]

            for c_id in children_id:
                for object_c in objs_data:
                    if object_c["objectId"] == c_id:
                        child = RearrangeObject.create(object_c, objs_data)
                        children[c_id] = child
                        break

        return cls(
            object_id,
            geometry=polys_2d,
            asset_id=asset_id,
            rotation=rotation,
            children=children,
            props=props,
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
class RearrangeWorld(World):
    EPSILON: float = 1e-3
    """Small value to compare floats within a bound."""

    @classmethod
    def create(cls, world_config):
        world_id = world_config.get("id", "RearrangeWorld")
        world = cls(world_id)
        world.grid_size = world_config["grid_size"]
        # asset_path = world_config["asset_path"]
        # assets = ai2thor_utils.load_assets(asset_path)
        world_data = world_config["data"]

        def get_room_polygons(objects, scene_corner_points):
            corner_points = scene_corner_points
            # for o in objects:
            #     bbox_cornerpoints = o.get("axisAlignedBoundingBox").get("cornerPoints")
            #     corner_points.extend(bbox_cornerpoints)
            minx = min([x[0] for x in corner_points])
            maxx = max([x[0] for x in corner_points])
            minz = min([x[2] for x in corner_points])
            maxz = max([x[2] for x in corner_points])
            return [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]

        room_polygons = get_room_polygons(
            world_data["objects"], world_data["sceneBounds"]["cornerPoints"]
        )
        # there is only one room
        world.add_room(RearrangeRoom.create("room|0", room_polygons))

        for i in range(len(room_polygons)):
            wall_polygon = [
                room_polygons[i],
                room_polygons[(i + 1) % len(room_polygons)],
                room_polygons[(i + 1) % len(room_polygons)],
                room_polygons[i],
            ]
            world.add_wall(RearrangeWall.create("wall" + str(i), wall_polygon))

        for object in world_data["objects"]:
            parentReceptacles = object["parentReceptacles"]
            parent = []
            if parentReceptacles is not None:
                for r in parentReceptacles:
                    if "Floor|" in r:
                        continue
                    else:
                        parent.append(r)
            if "Floor|" not in object["objectId"] and len(parent) == 0:
                world.add_object(RearrangeObject.create(object, world_data["objects"]))
        world.room_polygons = room_polygons
        logger.info(f"Successfully created world: {world_config['id']}")
        for obj_id, obj in world.objects.items():
            world.id2object[obj_id] = obj
            children = obj.find_all_children()
            for child in children:
                world.id2object[child.id] = child
        return world

    def add_room(self, room: RearrangeRoom) -> Optional[str]:
        self.rooms[room.id] = room
        return room.id

    def add_wall(self, wall: RearrangeWall) -> Optional[str]:
        logger.debug(wall.room2room)
        if len(wall.room2room) > 0:
            for r in wall.room2room:
                if r in self.rooms:
                    if self.rooms.get(r).add_wall(wall) is None:
                        return None

        self.walls[wall.id] = wall
        return wall.id

    def add_object(self, object: RearrangeObject):
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
