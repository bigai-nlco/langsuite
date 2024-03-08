from collections import defaultdict
import json
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from langsuite.shapes import Point2D, Polygon2D

import networkx as nx


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
