from cProfile import label
from collections import Counter, OrderedDict
from copy import deepcopy
from os import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from langsuite.utils.logging import logger
from overrides import override
from plotly.graph_objects import Figure
from shapely.geometry import Polygon

from langsuite.shapes import Geometry, Line2D, Point2D, Polygon2D, Vector2D
from shapely.geometry import MultiPoint
from langsuite.suit.exceptions import NotRegisteredError, ParameterMissingError
from langsuite.suit import Action, World
from langsuite.utils import math_utils
from langsuite.suit import WORLD_REGISTRY
from langsuite.worlds.basic2d_v0.physical_entity import (
    LocateAt,
    Object2D,
    PhysicalAgent,
    PhysicalEntity2D,
    Room2D,
)
import langsuite.worlds.basic2d_v0.utils as WUtils
import networkx as nx

@WORLD_REGISTRY.register()
class Basic2DWorld_V0(World):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.name: str = name
        self._objects: Dict[str, Object2D] = dict()
        self._obj_counter = Counter()
        self._object_id2index: Dict[str, str] = dict()
        self._object_index2id: Dict[str, str] = dict()
        self.ground = Room2D(
            room_id="_ground", geo_shape=Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])
        )
        self.rooms: Dict[str, Room2D] = {"_ground": self.ground}
        self._agents: Dict[str, PhysicalAgent] = dict()
        self.timestamp = -1
        self._observation: Dict[str, dict] = dict()

    @property
    @override
    def agents(self):
        return self._agents

    def get_object(self, oid: str) -> Object2D:
        try:
            return self._objects[oid]
        except KeyError as e:
            raise ParameterMissingError({"object": oid}) from e

    def _as_pos(self, obj_or_pos: Union[str, PhysicalEntity2D, Point2D]) -> Point2D:
        if isinstance(obj_or_pos, str):
            return self.get_object(obj_or_pos).position
        elif isinstance(obj_or_pos, PhysicalEntity2D):
            return obj_or_pos.position
        else:
            return obj_or_pos

    def get_room(self, obj_or_pos: Union[str, PhysicalEntity2D, Point2D]) -> Room2D:
        pos = self._as_pos(obj_or_pos)
        return next(
            (v for v in self.rooms.values() if v.geometry.contains(pos)), self.ground
        )

    def distance_to_pos(
        self,
        src: Union[str, PhysicalEntity2D, Point2D],
        dest: Union[str, PhysicalEntity2D, Point2D],
    ) -> Point2D:
        distance = self._as_pos(dest) - self._as_pos(src)
        return distance

    def intersects(self, traj, collect=True) -> Tuple[bool, List[str]]:
        def iter_intersects(entity: PhysicalEntity2D, traj: Geometry):
            # if entity is not self.ground:
            for child in entity.inventory:
                if iter_intersects(child, traj):
                    return True
            return entity != self.ground and entity.intersects(traj)

        def iter_intersects_with(entity: PhysicalEntity2D, traj: Geometry):
            for child in entity.inventory:
                yield from iter_intersects_with(child, traj)
            if entity.intersects(traj):
                yield entity.name

        if collect:
            result = list(iter_intersects_with(self.ground, traj))
            return (len(result) > 0, result)
        else:
            return (iter_intersects(self.ground, traj), list())

    # TODO: I just copied the logic, need test.
    def is_valid_trajectory(self, traj: Union[Point2D, Line2D]) -> bool:
        # XXX: follows the orginal logic, but why was so? I need to check this later.
        if isinstance(traj, Point2D):
            traj = Line2D([traj, Point2D(traj.x + 1, traj.y + 1)])

        if len(traj.coords) < 2:
            # XXX: Why?
            return True
        elif len(traj.coords) == 2:
            return not self.intersects(traj, collect=True)[0]
        else:
            for i, coord in enumerate(traj.coords[:-1]):
                # WHAT?
                # segment = Line2D([coord, traj[i + 1]])
                segment = Line2D([coord, traj.coords[i + 1]])
                if not self.is_valid_trajectory(segment):
                    return False
            return True

    def can_observe(self, agent: Union[str, PhysicalAgent], obj: PhysicalEntity2D):
        def nested_find(d: dict, t: str):
            for v in d.values():
                if v == t:
                    return True
                elif isinstance(v, list) and any(nested_find(s_d, t) for s_d in v):
                    return True
                elif isinstance(v, dict) and nested_find(v, t):
                    return True
            return False

        if isinstance(agent, PhysicalAgent):
            agent = agent.name
        return nested_find(self._observation[agent], self.object_id2index(obj.name))

    def exists_in_obs(
        self,
        agent: Union[str, PhysicalAgent],
        obj_type: str,
        cond_attr: str,
        exp_val: object,
    ):
        def nested_find(d: Union[dict, list]):
            if isinstance(d, dict):
                if d.get("type") == obj_type and (
                    exp_val is None or d.get(cond_attr) == exp_val
                ):
                    return True
                content = d.get("content")
                if content:
                    return nested_find(content)
            elif isinstance(d, list) and any(nested_find(sd) for sd in d):
                return True
            return False

        if isinstance(agent, PhysicalAgent):
            agent = agent.name
        return any(nested_find(l) for l in self._observation[agent].values())

    def object_id2index(self, oid: str) -> str:
        if not oid in self._object_id2index:
            obj = self._objects[oid]
            nid = self._obj_counter[obj.obj_type]
            self._object_id2index[oid] = f"{obj.obj_type}_{nid}".lower()
            self._object_index2id[self._object_id2index[oid]] = oid
            self._obj_counter[obj.obj_type] += 1
        return self._object_id2index[oid]

    def object_index2id(self, oid: str) -> str:
        return self._object_index2id[oid]

    def make_id_list(self, objects: Iterable[PhysicalEntity2D]) -> str:
        return ",".join(sorted(map(lambda x: self.object_id2index(x.name), objects)))

    def _in_vision(self, agent: PhysicalAgent, entity: PhysicalEntity2D) -> bool:
        geometry = entity.geometry
        if isinstance(geometry, Polygon2D):
            return agent.view_geometry.intersects(geometry)
        if isinstance(geometry, Point2D):
            return agent.view_geometry.contains(geometry)
        return False

    def _iter_get_independent_objects(
        self, entity: PhysicalEntity2D, agent: Optional[PhysicalAgent]
    ):
        if not isinstance(entity, Room2D):
            if (agent is None) or self._in_vision(agent, entity):
                yield entity
        if not getattr(entity, "receptacle", False):
            for child in entity.inventory:
                yield from self._iter_get_independent_objects(child, agent)

    # XXX is inherit right?
    def iter_expand2index(
        self, entity: PhysicalEntity2D, agent: Optional[PhysicalAgent], inherit_pos=False
    ) -> Optional[Dict[str, object]]:
        if (agent is not None) and not (inherit_pos or self._in_vision(agent, entity)):
            return None
        assert hasattr(entity, "receptacle")
        if entity.receptacle:  # type: ignore
            content = [self.iter_expand2index(child, agent) for child in entity.inventory]
            return {
                "index": self.object_id2index(entity.name),
                "content": list(filter(lambda x: x is not None, content)),
                **entity.list_textual_attrs(),
            }
        else:
            return {
                "index": self.object_id2index(entity.name),
                **entity.list_textual_attrs(),
            }

    def _chain_expand2index(self, agent, l) -> Sequence[Dict[str, object]]:
        """l are pre-collected that can't be none"""
        return list(map(lambda x: self.iter_expand2index(x, agent), l))  # type: ignore

    def describe_all(self) -> Sequence[Dict[str, object]]:
        objects = self._iter_get_independent_objects(self.ground, agent=None)
        return self._chain_expand2index(agent=None, l=objects)

    # TODO more type of obs
    @override
    def update(self):
        self.timestamp += 1
        for agent in self._agents.values():
            observed_objects = list(
                self._iter_get_independent_objects(self.ground, agent)
            )

            middle_objs = []
            left_objs = []
            right_objs = []

            middle_point = agent.position + agent.rotation * agent.max_view_distance
            middle_line = Line2D([agent.position, middle_point])
            left_line = deepcopy(middle_line)
            left_line.rotate(-agent.view_angle / 2)
            right_line = deepcopy(middle_line)
            right_line.rotate(agent.view_angle / 2)

            for obj in observed_objects:
                distance_dict = {
                    "middle_distance": obj.geometry.shapely_geo.distance(
                        middle_line.shapely_geo
                    ),
                    "left_distance": obj.geometry.shapely_geo.distance(
                        left_line.shapely_geo
                    ),
                    "right_distance": obj.geometry.shapely_geo.distance(
                        right_line.shapely_geo
                    ),
                }
                min_dis = sorted(distance_dict.items(), key=lambda dis: dis[1])
                if min_dis[0][0] == "middle_distance":
                    middle_objs.append(obj)
                elif min_dis[0][0] == "left_distance":
                    left_objs.append(obj)
                elif min_dis[0][0] == "right_distance":
                    right_objs.append(obj)


            self._observation[agent.name] = {
                "middle_objs": self._chain_expand2index(agent, middle_objs),
                "left_objs": self._chain_expand2index(agent, left_objs),
                "right_objs": self._chain_expand2index(agent, right_objs),
            }

    @override
    def get_observation(
        self, agent_name: str
    ) -> Dict[str, Sequence[Dict[str, object]]]:
        return self._observation[agent_name]

    @override
    def step(
        self, agent_name: str, action_dict: dict
    ) -> Tuple[bool, Dict[str, object]]:
#        print(action_dict)
        type_str = action_dict.pop("action")
        for k, v in list(action_dict.items()):
            if k.endswith('_index'):
                new_key = f'{k[:-6]}_id'
                action_dict[new_key] = self.object_index2id(v)
                action_dict.pop(k)
        action_dict["agent"] = self.agents[agent_name]
        action_dict["world"] = self
        try:
            action_type = self.action_reg.get(type_str)
            action: Action = action_type(**action_dict)
        except KeyError as e:
            raise NotRegisteredError({"action": type_str}) from e
        result = action.exec()
        if getattr(action, "is_slice", False):
            obj_index = result[1]['object']
            self.replace_sliced(obj_index)
        return result

    def replace_sliced(self, obj_index: str):
        obj_id = self.object_index2id(obj_index)
        obj = self._objects.pop(obj_id)
        obj._locate_at.receptacle.remove_from_inventory(obj)
        
        if obj.obj_type == "Egg":
            new_obj_type = "EggCracked"
        else:
            new_obj_type = f"{obj.obj_type}Sliced"

        #TODO sliced pieces could have different pos and size
        #Cracked Egg are not pieces.
        SLICED_PIECES = 1 if new_obj_type == "EggCracked" else 10
        for i in range(SLICED_PIECES):
            new_obj = obj.copy()
            assert new_obj._locate_at.receptacle == obj._locate_at.receptacle
            new_obj.obj_type = new_obj_type
            new_obj.obj_name = f"{obj.name}|{new_obj_type}_{i+1}"
            setattr(new_obj, "isCooked", getattr(obj, "isCooked", False))
            assert new_obj._locate_at.receptacle == obj._locate_at.receptacle
            new_obj._locate_at.receptacle.add_to_inventory(new_obj)
            assert new_obj in obj._locate_at.receptacle.inventory
            self._objects[new_obj.name] = new_obj

    @override
    def render(self) -> Figure:
        fig = Figure()
        for room in self.rooms.values():
            room.render(fig, label = room.name)
        #TODO render windows and doors --- if they are implemented.
        for obj in self._objects.values():
            obj.render(fig, label = self.object_id2index(obj.name))
        for agent in self.agents.values():
            agent.render(fig, label = agent.name)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(showlegend=False)
        fig.show()
        return fig

    @classmethod
    def create(cls, world_data, fmt="PROC_THOR"):
        # TODO more create way
        if fmt == "PROC_THOR":
            return cls.create_from_ProcThor(world_data)
        else:
            raise NotImplementedError()

    @classmethod
    def create_from_ProcThor(
        cls, world_data, meta_data=WUtils.DataLoader.get_thor_database()
    ):
        world = Basic2DWorld_V0(world_data)

        # Rooms start
        for room_data in world_data["rooms"]:
            geo_raw = room_data["floorPolygon"]
            geo_shape = MultiPoint([WUtils.toTuple2D(p) for p in geo_raw]).convex_hull
            r = Room2D(
                room_id=room_data["id"],
                geo_shape=geo_shape,
            )
            world.ground.inventory.add(r)
            world.rooms[r.room_id] = r

        for door_data in world_data["doors"]:
            r0 = door_data.get("room0")
            if r0 is not None:
                world.rooms[r0].set_door()
            r1 = door_data.get("room1")
            if r1 is not None:
                world.rooms[r1].set_door()
            if r0 is None and r1 is None:
                # TODO error log
                pass

        for window_data in world_data["windows"]:
            r0 = window_data.get("room0")
            if r0 is not None:
                world.rooms[r0].set_window()
            r1 = window_data.get("room1")
            if r1 is not None:
                world.rooms[r1].set_window()
            if r0 is None and r1 is None:
                # TODO error log
                pass
        # Rooms done.

        # Objects start.
        for object_info in world_data["objects"]:
            cls.iter_create_object(
                object_info=object_info,
                assets_data=meta_data,
                world=world,
                location=None,
                obj_collector=world._objects,
            )

        # TODO HACK faucet -> basin, knob -> burner ?
        def locate_by_matching(X: Iterable[Object2D], Y: Iterable[Object2D]):
            DIST_THRESHOLD = 0.5
            edges = [(x.obj_name, y.obj_name, (x.position - y.position).modulus) for x in X for y in Y]
            edges = list(filter(lambda x: x[2] < DIST_THRESHOLD, edges))
            for x, y in WUtils.compute_matching(edges):
                ox = world._objects[x]
                oy = world._objects[y]
                rel = LocateAt(oy, ox._timestamp, ox.position - oy.position)
                ox.update_position(rel)

        #HACK locate faucets to closest basins.
        faucets = filter(lambda x: x.obj_type == 'Faucet', world._objects.values())
        basins = filter(lambda x: x.obj_type.endswith('Basin'), world._objects.values())
        locate_by_matching(faucets, basins)

        #HACK locate stove knobs, it will be better to read their relation from THOR (if possible)
        knobs = filter(lambda x: x.obj_type == 'StoveKnob', world._objects.values())
        burners = filter(lambda x: x.obj_type == 'StoveBurners', world._objects.values())
        locate_by_matching(knobs, burners)
        # Objects done.

        # Agents start.
        agents_data = world_data["metadata"]["agent"]
        task_spec_agents_cfg = world_data["agents"]

        def split(cfg):
            for i, data in enumerate(cfg):
                ag_id = data.get("agentId", f"agent_{i}")
                yield (ag_id, data)

        task_spec_agents_cfg = {x: y for x, y in split(task_spec_agents_cfg)}

        # If single agent
        if isinstance(agents_data, dict):
            if "agentId" not in agents_data:
                agents_data["agentId"] = "agent_0"
            agents_data = [agents_data]
        # TODO more agent parameters?
        for ag_data in agents_data:
            agent_id = ag_data["agentId"]
            task_cfg = task_spec_agents_cfg[agent_id]
#            ag_aov = task_cfg.get("view_angle", ag_data.get("view_angle"))
#            if ag_aov is None:
            ag_fl = task_cfg.get("focal_length")
            
            ag_rot = Vector2D(0, 1)
            ag_rot.rotate(ag_data["rotation"])
            pa = PhysicalAgent(
                agent_id=agent_id,
                position=Point2D(*WUtils.toTuple2D(ag_data["position"])),
                rotation=ag_rot,
                step_size=task_cfg.get("step_size", ag_data.get("step_size")),
                max_view_distance=task_cfg.get(
                    "max_view_distance", ag_data.get("max_view_distance")
                ),
                focal_length=ag_fl,
                max_manipulate_distance=task_cfg.get(
                    "max_manipulate_distance", ag_data.get("max_manipulate_distance")
                ),
                inventory_capacity=task_cfg.get(
                    "inventory_capacity", ag_data.get("inventory_capacity")
                ),
            )
            world.agents[agent_id] = pa
        # Agents done.
        logger.debug(world._objects.keys())
        world.update()
        return world

    # TODO add log
    @classmethod
    def _bbox_hack(cls, pos_or_shape: Union[Point2D, list]):
        if isinstance(pos_or_shape, Point2D):
            offsets = [
                Point2D(-0.05, 0.05),
                Point2D(0.05, 0.05),
                Point2D(0.05, -0.05),
                Point2D(-0.05, -0.05),
            ]
            return map(lambda o: pos_or_shape + o, offsets)
        else:
            raw_shape = list(map(WUtils.toTuple2D, pos_or_shape))
            polygon = Polygon2D(raw_shape)
            return [
                (polygon.x_min, polygon.y_min),
                (polygon.x_max, polygon.y_min),
                (polygon.x_max, polygon.y_max),
                (polygon.x_min, polygon.y_max),
            ]

    @classmethod
    def iter_create_object(
        cls,
        object_info,
        assets_data,
        world: "Basic2DWorld_V0",
        location,
        obj_collector: dict,
    ):
        # print(object_info)
        assets_id = object_info["assetId"]
        obj_id = object_info["objectId"]
        # use center instead of real pos is for using axis aligned bbox, don't change it.
        if "axisAlignedBoundingBox" in object_info:
            static_bbox = object_info["axisAlignedBoundingBox"]["cornerPoints"]
            obj_pos = Point2D(
                *WUtils.toTuple2D(object_info["axisAlignedBoundingBox"]["center"])
            )
        elif assets_id in assets_data:
            assert "axisAlignedBoundingBox" in assets_data[assets_id]
            axis_data = assets_data[assets_id]["axisAlignedBoundingBox"]
            static_bbox = axis_data["cornerPoints"]
            obj_pos = Point2D(*WUtils.toTuple2D(axis_data["center"]))
        else:  # do some hack
            obj_pos = Point2D(*WUtils.toTuple2D(object_info["position"]))
            if "objectOrientedBoundingBox" in object_info:
                static_bbox = cls._bbox_hack(
                    object_info["objectOrientedBoundingBox"]["cornerPoints"]
                )
                obj_pos = Point2D(*WUtils.toTuple2D(object_info["position"]))
            # HACK use atomic BBox instead
            static_bbox = cls._bbox_hack(obj_pos)

        if location is None:
            location = world.get_room(obj_pos)
        locate = LocateAt(
            location, world.timestamp, world.distance_to_pos(location, obj_pos)
        )
        obj_rot = Vector2D(*WUtils.toTuple2D(object_info["rotation"]))

        geo_shape = [WUtils.toTuple2D(p) for p in static_bbox]
        geo_shape = MultiPoint(geo_shape).convex_hull

        attrs = dict()
        for key in WUtils.DataLoader.get_raw_attr_names():
            if key in object_info:
                attrs[key] = object_info[key]
            elif assets_id in assets_data:
                attrs[key] = assets_data[assets_id][key]
            # need validation!!! Do IDs act as identifiers cross datasets?
            # elif obj_id in assets_data:
            #    attrs[key] = assets_data[obj_id][key]
        o = Object2D(
            obj_id=obj_id,
            obj_type=object_info["objectType"],
            rotation=obj_rot,
            locate_at=locate,
            geometry=geo_shape,
            attrs=attrs,
        )
        children = object_info["children"]
        if children is not None:
            for child in children:
                cls.iter_create_object(
                    child, assets_data, world, o, obj_collector
                )
        obj_collector[obj_id] = o
