# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np

from langsuite.shapes import Vector2D


def md5_hash_str_as_int(to_hash: str):
    return int(
        hashlib.md5(to_hash.encode()).hexdigest(),
        16,
    )


def get_direction(rotation):
    rotation = rotation % 360
    if 0 <= rotation <= 30:
        return "north"
    elif 30 < rotation < 60:
        return "northeast"
    elif 60 <= rotation <= 120:
        return "east"
    elif 120 < rotation < 150:
        return "southeast"
    elif 150 <= rotation <= 210:
        return "south"
    elif 210 < rotation < 240:
        return "southwest"
    elif 240 <= rotation <= 300:
        return "west"
    elif 300 < rotation < 330:
        return "northwest"
    elif 330 <= rotation <= 360:
        return "north"


def get_direction_vec(direction):
    if direction == "north":
        return Vector2D(0, 1)
    elif direction == "northeast":
        return Vector2D(1, 1)
    elif direction == "east":
        return Vector2D(1, 0)
    elif direction == "southeast":
        return Vector2D(1, -1)
    elif direction == "south":
        return Vector2D(0, -1)
    elif direction == "southwest":
        return Vector2D(-1, -1)
    elif direction == "west":
        return Vector2D(-1, 0)
    elif direction == "northwest":
        return Vector2D(-1, 1)


@staticmethod
def position_dist(
    p0: Mapping[str, Any],
    p1: Mapping[str, Any],
    l1_dist: bool = False,
) -> float:
    """Distance between two points of the form {"x": x, "z":z"}."""
    if l1_dist:
        return abs(p0["x"] - p1["x"]) + abs(p0["z"] - p1["z"])
    else:
        return math.sqrt((p0["x"] - p1["x"]) ** 2 + (p0["z"] - p1["z"]) ** 2)


@staticmethod
def rotation_dist(a: float, b: float):
    """Distance between rotations."""

    def deg_dist(d0: float, d1: float):
        dist = (d0 - d1) % 360
        return min(dist, 360 - dist)

    return deg_dist(a, b)


@staticmethod
def angle_between_rotations(a: float, b: float):
    # TODO
    # return np.abs(
    #     (180 / (2 * math.pi))
    #     * (
    #         Rotation.from_euler("xyz", [a[k] for k in "xyz"], degrees=True)
    #         * Rotation.from_euler("xyz", [b[k] for k in "xyz"], degrees=True).inv()
    #     ).as_rotvec()
    # ).sum()
    return np.abs(a - b)


@staticmethod
def obj_list_to_obj_name_to_pose_dict(objects: List[Dict[str, Any]]) -> OrderedDict:
    """Helper function to transform a list of object data dicts into a
    dictionary."""
    objects = [o for o in objects]
    d = OrderedDict((o["name"], o) for o in sorted(objects, key=lambda x: x["name"]))
    # assert len(d) == len(objects)
    return d


def get_pose_info(
    objs: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Return data about each specified object.

    For each object, the return consists of its type, position,
    rotation, openness, and bounding box.
    """
    # list of objects
    if isinstance(objs, Sequence):
        return [extract_obj_data(obj) for obj in objs]
    # single object
    return extract_obj_data(objs)


def extract_obj_data(obj):
    """Return object evaluation metrics based on the env state."""
    return {
        "type": obj["objectType"],
        "position": obj["position"],
        "rotation": obj["rotation"],
        "openness": obj["openness"] if obj["openable"] else None,
        "pickupable": obj["pickupable"],
        "broken": obj["isBroken"],
        "objectId": obj["objectId"],
        "name": obj["name"],
        "parentReceptacles": None,
        "bounding_box": None,
    }
