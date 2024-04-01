from collections import defaultdict
import copy
from datetime import datetime

import json
from pathlib import Path
from typing import Dict, List, Tuple, Type
from overrides import override
from shapely import Point
from langsuite import utils

from langsuite.shapes import Vector2D
from langsuite.suit import LangSuiteEnv
from langsuite.suit.message import MessageHandler
from langsuite.suit.task import LangsuiteTask, TaskAction, TaskStatus
from langsuite.suit import TASK_REGISTRY
from langsuite.tasks.alfred_v0.actions import (
    CloseObject,
    DropObject,
    MoveAhead,
    OpenObject,
    PickupObject,
    PutObject,
    SliceObject,
    Stop,
    ToggleObjectOff,
    ToggleObjectOn,
    RotateLeft,
    RotateRight,
)
from langsuite.tasks.alfred_v0.message_handler import AlfredHandler, AlfredStatus
from langsuite.utils import template_utils, logging
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0
import langsuite.worlds.basic2d_v0.utils as WUtils


@TASK_REGISTRY.register()
class AlfredTask_V0(LangsuiteTask):
    world_type = Basic2DWorld_V0.__name__

    ACTIONS: List[Type[TaskAction]] = [
        MoveAhead,
        RotateLeft,
        RotateRight,
        PickupObject,
        DropObject,
        PutObject,
        OpenObject,
        CloseObject,
        ToggleObjectOn,
        ToggleObjectOff,
        SliceObject,
        Stop,
    ]

    def __init__(self, task_data, task_cfg) -> None:
        super().__init__(task_data, task_cfg)

    @override
    def make_status(self, task_data) -> TaskStatus:
        return AlfredStatus(
            mrecep_target=task_data["target_status"]["mrecep_target"],
            object_sliced=task_data["target_status"]["object_sliced"],
            object_target=task_data["target_status"]["object_target"],
            parent_target=task_data["target_status"]["parent_target"],
            toggle_target=task_data["target_status"]["toggle_target"],
        )

    @override
    def make_handler(self) -> MessageHandler:
        name_mapping = {x.name: x.__name__ for x in self.ACTIONS}
        return AlfredHandler(self.task_type, self.target_status, name_mapping)

    @classmethod
    @override
    def create(
        cls, task_cfg, task_data=None, cmd_cli=None, web_ui=None
    ) -> LangsuiteTask:
        if task_data is None:
            path = "./data/alfred"
            tasks = cls.load_data(path, "test")
            task_data = tasks[100]
            logging.logger.debug(task_data["path"])

        return super().create(task_cfg, task_data, cmd_cli, web_ui)

    @classmethod
    def load_data(cls, data_dir, stage):
        tasks = []
        dev_path = Path(data_dir, stage + ".json")
        with open(
            dev_path,
            "r",
            encoding="utf-8",
        ) as data_f:
            dev_data = json.load(data_f)
        # dev_data = {"a": [1] * 11}
        for task_type, task_paths in dev_data.items():
            for task_path in task_paths:
                with open(
                    Path(data_dir).joinpath(task_path),
                    "r",
                    encoding="utf-8",
                ) as scene_f:
                    task_json = json.load(scene_f)
                    task_json["path"] = task_path
                    tasks.append(task_json)
                    logging.logger.debug('loaded: %s', task_path)
        return tasks

    @classmethod
    def _multi_receptacle_hack(cls, child, parents) -> str:
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

    @classmethod
    @override
    def _convert_task_data_format(cls, task_cfg, raw_task_data) -> dict:
        task_cfg["task_description"] = raw_task_data["turk_annotations"]["anns"][0][
            "task_desc"
        ]
        task_cfg["task_type"] = raw_task_data["task_type"]
        task_cfg["target_status"] = raw_task_data["pddl_params"]

        scence_data = raw_task_data["scene"]
        world_data = dict()

        # convert room
        rooms = []
        r_id = scence_data["floor_plan"]
        geo_raw = scence_data["polys_2d"]
        rooms.append({"id": r_id, "floorPolygon": geo_raw})
        world_data["rooms"] = rooms

        # convert object
        doors = []
        windows = []
        objects = dict()
        world_data["objects"] = []
        receptacles_rels = dict()

        for object_data in scence_data["objects"]:
            if object_data['objectType'] == 'Floor':
                continue
            object_data["assetId"] = object_data.pop("name")
            if object_data["objectBounds"]:
                bounding_box = object_data.pop("objectBounds")
                corner_points = bounding_box.pop("objectBoundsCorners")
                object_data["axisAlignedBoundingBox"] = {
                    "cornerPoints": corner_points,
                    "center": object_data["position"],
                }

            object_data["rotation"] = {"x": 0.0, "y": 0.0, "z": 0.0}
            obj_id = object_data["objectId"]
            if object_data["parentReceptacles"] is not None:
                receptacles_rels[obj_id] = object_data["parentReceptacles"]
            objects[obj_id] = object_data
            object_data["children"] = list()
            object_data["isHeatSource"] = object_data.pop("canChangeTempToHot")
            object_data["isColdSource"] = object_data.pop("canChangeTempToCold")
            object_data["temperature"] = object_data.pop("ObjectTemperature")
            if object_data["objectType"] == "Window":
                windows.append(object_data)
            elif object_data["objectType"] == "Door":
                doors.append(object_data)

        world_data["objects"], _ = WUtils.receptacles_hack(receptacles_rels, objects)

        world_data["doors"] = doors
        world_data["windows"] = windows

        ag_init = scence_data["init_action"]
        ag_rot = ag_init["rotation"]
        ag_pos = {"x": ag_init["x"], "z": ag_init["z"]}
              
        agents_data = {
            "position": ag_pos,
            "rotation": ag_rot,
        }

        world_data["metadata"] = dict()
        world_data["metadata"]["agent"] = agents_data

        task_cfg["world_data"] = world_data
        task_cfg["world_type"] = cls.world_type

        name_dict = {x.__name__: x.name for x in cls.ACTIONS}
        expert_actions = []
        for raw_action in raw_task_data["plan"]["low_actions"]:
            if raw_action["api_action"]["action"] in name_dict:
                raw_action["action_name"] = name_dict[raw_action["api_action"]["action"]]
                expert_actions.append(raw_action)
        expert_actions.append({"api_action": {"action": "Stop"}, "action_name": "stop"})

        task_cfg["agents"][0]["expert_actions"] = expert_actions

        with open(task_cfg["template"], "r") as template_file:
            template = template_utils.split(json.load(template_file))

        task_cfg["agents"][0]["template"] = template

        return task_cfg
