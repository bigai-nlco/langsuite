from collections import defaultdict
import copy

import json
from pathlib import Path
from typing import Dict, List, Type
from overrides import override
from shapely import Point

from langsuite.shapes import Point2D
from langsuite.suit.message import MessageHandler
from langsuite.suit.task import LangsuiteTask, TaskAction, TaskStatus
from langsuite.suit import TASK_REGISTRY
from langsuite.tasks.rearrange_v0.actions import (
    DropObject,
    MoveAhead,
    OpenObject,
    CloseObject,
    PutObject,
    Stop,
    RotateLeft,
    RotateRight,
)
from langsuite.tasks.rearrange_v0.message_handler import (
    RearrangeStatus,
    RerrangeHandler,
)
from langsuite.tasks.rearrange_v0.utils import compute_square_hack
from langsuite.utils import logging, template_utils
import langsuite.worlds.basic2d_v0.utils as WUtils
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0


@TASK_REGISTRY.register()
class RearrangeTask_V0(LangsuiteTask):
    world_type = "Basic2DWorld_V0"

    ACTIONS: List[Type[TaskAction]] = [
        MoveAhead,
        RotateLeft,
        RotateRight,
        OpenObject,
        CloseObject,
        DropObject,
        PutObject,
        Stop,
    ]

    def __init__(self, task_data, task_cfg) -> None:
        task_data["task_description"] = ""
        task_data["task_type"] = "rearrange"
        super().__init__(task_data, task_cfg)

        world: Basic2DWorld_V0 = self.env.world  # type: ignore
        described_collector = set()
        self.task_description = RerrangeHandler.pack_list(
            world.describe_all(), "In the room", described_collector
        )
        self.target_status = self._make_status(task_data, world)

    @override
    def make_status(self, task_data) -> TaskStatus:
        """The status need to be computed with world APIs"""
        return TaskStatus()

    def _make_status(self, task_data, world: Basic2DWorld_V0) -> RearrangeStatus:
        change_rec, change_open, change_pos, target_rec, target_open, target_pos = task_data[
            "target_status"
        ]
        change_pos = (
            (k, compute_square_hack(world, v[0]), compute_square_hack(world, v[1]))
            for k, v in change_pos.items()
        )
        change_pos = {x[0]: x[1] for x in filter(lambda x: x[1] != x[2], change_pos)}
        target_pos = {k: compute_square_hack(world, v) for k, v in target_pos.items()}
        return RearrangeStatus(
            change_rec, change_open, change_pos, target_rec, target_open, target_pos
        )

    @override
    def make_handler(self) -> MessageHandler:
        name_mapping = {x.name: x.__name__ for x in self.ACTIONS}
        return RerrangeHandler(self.task_type, self.target_status, name_mapping)

    @classmethod
    @override
    def create(
        cls, task_cfg, task_data=None, cmd_cli=None, web_ui=None
    ) -> LangsuiteTask:
        if task_data is None:
            path = "./data/rearrange"
            tasks = cls.load_data(path, "test")
            task_data = tasks[100]
            logging.logger.debug(task_data["task_id"])

        return super().create(task_cfg, task_data, cmd_cli, web_ui)

    @classmethod
    def load_data(cls, data_dir, stage):
        rearr_data = open(Path(data_dir, f"rearrange_{stage}", f"{stage}.json"))

        tasks = []
        for data in rearr_data.readlines():
            task_json = json.loads(data)
            task_json["task_id"] = f"{task_json['scene']}_{task_json['index']}"
            tasks.append(task_json)
        return tasks

    @classmethod
    @override
    def _convert_task_data_format(cls, task_cfg, raw_task_data) -> dict:
        # task_cfg['task_description'] =
        # task_cfg['target_status'] =

        scence_data = raw_task_data["start_scene"]
        world_data = dict()

        # convert room
        rooms = []
        r_id = raw_task_data["scene"]
        geo_raw = scence_data["sceneBounds"]["cornerPoints"]
        rooms.append({"id": r_id, "floorPolygon": geo_raw})
        world_data["rooms"] = rooms

        # convert object
        original_objects = scence_data["objects"]
        converted = WUtils.ai2thor_obj_2_procthor(original_objects)
        world_data["objects"] = converted[0]
        world_data["doors"] = converted[1]
        world_data["windows"] = converted[2]
        original_dep = converted[3]

        ag_init = scence_data["agent"]
        ag_rot = ag_init["rotation"]["y"]
        ag_pos = {"x": ag_init["position"]["x"], "z": ag_init["position"]["z"]}

        agents_data = {"position": ag_pos, "rotation": ag_rot}

        world_data["metadata"] = dict()
        world_data["metadata"]["agent"] = agents_data

        task_cfg["world_data"] = world_data
        task_cfg["world_type"] = cls.world_type

        with open(task_cfg["template"], "r") as template_file:
            template = template_utils.split(json.load(template_file))

        task_cfg["agents"][0]["template"] = template

        target_objects = raw_task_data["end_scene"]["objects"]
        _, _, _, target_dep = WUtils.ai2thor_obj_2_procthor(target_objects)

        task_cfg["target_status"] = cls._compute_status(
            original_objects, original_dep, target_objects, target_dep
        )

        return task_cfg

    # TODO compute not only dep. but also pos.
    @classmethod
    def _compute_status(
        cls, original_objects, original_dep, target_objects, target_dep
    ):
        tar_objs = {x["objectId"]: x for x in target_objects}

        change_rec = {}
        change_open = {}
        change_pos = {}

        target_open = {}
        target_pos = {}

        for obj in original_objects:
            if obj['objectType'] == 'Floor':
                continue
            name = obj["objectId"]
            if original_dep.get(name, None) != target_dep.get(name, None):
                change_rec[name] = original_dep.get(name, None)
            tar_obj = tar_objs[name]
            if obj['openable'] != tar_obj['openable'] or obj["isOpen"] != tar_obj["isOpen"]:
                change_open[name] = obj["isOpen"] if obj['openable'] else None
            target_open[name] = tar_obj["isOpen"] if tar_obj['openable'] else None
            obj_pos = Point2D(WUtils.toTuple2D(obj["axisAlignedBoundingBox"]["center"]))
            tar_pos = Point2D(
                WUtils.toTuple2D(tar_obj["axisAlignedBoundingBox"]["center"])
            )
            target_pos[name] = tar_pos
            if obj_pos.shapely_geo.distance(tar_pos.shapely_geo) > 0.05:
                change_pos[name] = (obj_pos, tar_pos)

        return (change_rec, change_open, change_pos, target_dep, target_open, target_pos)
