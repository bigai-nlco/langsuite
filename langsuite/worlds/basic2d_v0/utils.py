from collections import defaultdict
import json
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from langsuite.shapes import Point2D, Polygon2D
from shapely import Point

import networkx as nx

from langsuite.utils.logging import logger


class DataLoader:
    # HACK hack?
    _path = Path(__file__).parent.parent.parent.parent

    @classmethod
    def get_thor_database(cls):
        if not hasattr(cls, "_thor_data"):
            # FIXME use relative path
            with open(f"{cls._path}/data/assets.json", "r") as f:
                _database = json.load(f)
                cls._thor_data = dict()
                for data in _database:
                    assets_id = data["assetId"]
                    cls._thor_data[assets_id] = data
                    # XXX IS IT RIGHT?
                    obj_id = data["objectId"]
                    cls._thor_data[obj_id] = data
        return cls._thor_data

    @classmethod
    def get_object_rules(cls):
        if not hasattr(cls, "_object_rules"):
            # FIXME use a corrrect path
            with open(f"{cls._path}/data/assets_rules.json", "r") as f:
                cls._object_rules = json.load(f)
        return cls._object_rules

    @classmethod
    def get_raw_attr_names(cls):
        if not hasattr(cls, "_raw_attr_names"):
            data = cls.get_object_rules()["attr"]
            cls._raw_attr_names = []
            for attr in data:
                cls._raw_attr_names.append(attr["name"])
                if "premise" in attr:
                    cls._raw_attr_names.append(attr["premise"])
        return cls._raw_attr_names


def ai2thor_obj_2_procthor(org_objects):
    """
    Convert ai2thor objects to procthor objects.

    :param org_objects: ai2thor objects
    :type org_objects: list
    :rtype: tuple (objects, doors, windows, dependencies)
    """
    objects = dict()
    receptacles_rels = dict()
    doors = []
    windows = []

    for object_data in org_objects:
        if object_data["objectType"] == "Floor":
            continue
        object_data["assetId"] = object_data.pop("name")
        object_data["rotation"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        obj_id = object_data["objectId"]
        if object_data["parentReceptacles"] is not None:
            receptacles_rels[obj_id] = object_data["parentReceptacles"]
        objects[obj_id] = object_data
        object_data["children"] = list()

    if object_data["objectType"] == "Window":
        windows.append(object_data)
    elif object_data["objectType"] == "Door":
        doors.append(object_data)

    objects, dep = receptacles_hack(receptacles_rels, objects)
    dep = {x[0]: x[1] for x in dep}

    return objects, doors, windows, dep


def receptacles_hack(
    old_deps: Dict[str, list], flat_objects: Dict[str, Dict]
) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """
    Conver flat_objects with alfred-style rels into nested.

    :param old_rels: raw receptacle rels, child -> parents, use objectId as keys and values.
    :type old_rels: dict
    :param flat_objects: raw objects, key = objectId. !!! will modify this variable.
    :type flat_objects: dict
    :rtype: nested_objs, deps # format is adopted from procthor.
    """
    dependency = []
    for obj in old_deps:
        # Floor is room, not object (provided to LLM)
        parents = list(filter(lambda x: not x.startswith("Floor|"), old_deps[obj]))
        if len(parents) > 1:
            obj_data = flat_objects[obj]
            parents_data = map(lambda x: flat_objects[x], parents)
            receptacle = _multi_receptacle_hack(obj_data, parents_data)
        elif len(parents) == 1:
            receptacle = parents[0]
        else:  # len == 0
            continue
        dependency.append((obj, receptacle))
    order = ordering_objects(dependency, flat_objects.keys())
    logger.debug("sorted obj_id: %s", order)
    dep_map = defaultdict(list)
    for obj, rec in dependency:
        dep_map[rec].append(obj)
    logger.debug("dependencies: %s", dep_map)
    for rec in order:
        for obj in dep_map[rec]:
            flat_objects[rec]["children"].append(flat_objects.pop(obj))
    return list(flat_objects.values()), dependency


def _multi_receptacle_hack(child, parents) -> str:
    """
    Find the 'real' parent, hacked as the closet one.

    :param child: child data
    :type child: dict
    :param parents: parent data
    :type parents: dict
    :rtype: str #the objectId of parent
    """

    def get_pos_and_name(d):
        name = d["objectId"]
        pos = Point(d["position"]["x"], d["position"]["x"], d["position"]["z"])
        return (name, pos)

    c_info = get_pos_and_name(child)
    p_infos = map(get_pos_and_name, parents)
    return min(p_infos, key=lambda x: c_info[1].distance(x[1]))[0]


# TODO add format check
# XXX really bad and dangerous practice, why not unifoms data formats carefully?
def toTuple2D(p3d) -> Tuple[float, float]:
    if isinstance(p3d, Point2D):
        return (p3d.x, p3d.y)
    elif isinstance(p3d, dict):
        return (p3d["x"], p3d["z"])
    elif isinstance(p3d, list) and len(p3d) == 3:
        return (p3d[0], p3d[2])
    elif isinstance(p3d, list) and len(p3d) == 2:
        return (p3d[0], p3d[1])
    else:
        raise ValueError("Illegal format of point: ", p3d)


def ordering_objects(
    dependencies: Sequence[Tuple[str, str]], vertices: Optional[Iterable[str]] = None
) -> List[str]:
    """
    Return the order upon dependencies, if A depends on B, A should be ordered before B.

    :param dependencies: dependency edges
    :param vertices: name of vertices, default is None (read vertices from edges)
    :return: ordered vertices
    """
    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    G.add_edges_from(dependencies)
    order = list(nx.topological_sort(G))
    return order


def compute_matching(edges: Sequence[Tuple[str, str, float]]) -> List[Tuple[str, str]]:
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    matchings = nx.min_weight_matching(G)
    # HACK
    check_set = set(map(lambda x: f"{x[0]}__{x[1]}", edges))
    matchings = [
        (x[0], x[1]) if f"{x[0]}__{x[1]}" in check_set else (x[1], x[0])
        for x in matchings
    ]
    return matchings
