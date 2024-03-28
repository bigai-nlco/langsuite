from collections import defaultdict
import copy

import json
from pathlib import Path
from typing import Dict, List, Type
from overrides import override
from shapely import Point

from langsuite.suit.message import MessageHandler
from langsuite.suit.task import LangsuiteTask, TaskAction, TaskStatus
from langsuite.suit import TASK_REGISTRY
from langsuite.tasks.iqa_v0.actions import (
    MoveAhead,
    OpenObject,
    Stop,
    RotateLeft,
    RotateRight,
)
from langsuite.tasks.iqa_v0.message_handler import IQAHandler, IQAStatus
from langsuite.utils import logging, template_utils
import langsuite.worlds.basic2d_v0.utils as WUtils


@TASK_REGISTRY.register()
class IQATask_V0(LangsuiteTask):
    world_type = "Basic2DWorld_V0"

    ACTIONS: List[Type[TaskAction]] = [
        MoveAhead,
        RotateLeft,
        RotateRight,
        OpenObject,
        Stop,
    ]

    def __init__(self, task_data, task_cfg) -> None:
        super().__init__(task_data, task_cfg)

    @override
    def make_status(self, task_data) -> TaskStatus:
        return IQAStatus(
            answer=task_data['target_status']['answer'],
            object_class=task_data['target_status']['object_class'],
            recept=task_data['target_status'].get('recept')
        )

    @override
    def make_handler(self) -> MessageHandler:
        name_mapping = {x.name: x.__name__ for x in self.ACTIONS}
        return IQAHandler(self.task_type, self.target_status, name_mapping)

    @classmethod
    @override
    def create(
        cls, task_cfg, task_data=None, cmd_cli=None, web_ui=None
    ) -> LangsuiteTask:
        if task_data is None:
            path = "./data/iqa"
            tasks = cls.load_data(path, "test")
            task_data = tasks[100]
            logging.logger.debug(task_data["task_id"])

        return super().create(task_cfg, task_data, cmd_cli, web_ui)

    @classmethod
    def load_data(cls, data_dir, stage):
        # FIXME support different stages
        iqa_data = json.load(open(Path(data_dir, "iqa_test", "iqa_test_1k.json")))
        # iqa_data = json.load(open(Path(data_dir, "data", "iqa", "iqa_list_qa_counts_300.json")))

        tasks = []
        for _id, world_data in enumerate(iqa_data):
            task_json = world_data[0]
            task_json['target_status'] = world_data[1]
            task_json['task_id'] = _id
            tasks.append(task_json)
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
    def _receptacles_hack(
        cls, old_deps: Dict[str, str], flat_objects: Dict[str, Dict]
    ) -> List[dict]:
        """
        Conver flat_objects with alfread-style rels into nested.

        :param old_rels: raw receptacle rels, child -> parents, use objectId as keys and values.
        :type old_rels: dict
        :param flat_objects: raw objects, key = objectId. !!! will modify this variable.
        :type flat_objects: dict
        :rtype: list # format is adopted from procthor.
        """
        dependency = []
        for obj in old_deps:
            if len(old_deps[obj]) > 1:
                obj_data = flat_objects[obj]
                parents_data = map(lambda x: flat_objects[x], old_deps[obj])
                receptacle = cls._multi_receptacle_hack(obj_data, parents_data)
            elif len(old_deps[obj]) == 1:
                receptacle = old_deps[obj][0]
            else:  # len == 0
                continue
            dependency.append((obj, receptacle))
        order = WUtils.ordering_objects(dependency, flat_objects.keys())
        logging.logger.debug("sorted obj_id: %s", order)
        dep_map = defaultdict(list)
        for obj, rec in dependency:
            dep_map[rec].append(obj)
        logging.logger.debug("dependencies: %s", dep_map)
        for rec in order:
            # HACK do not need Floor
            if flat_objects[rec]["objectType"] == 'Floor':
                flat_objects.pop(rec)
                continue
            for obj in dep_map[rec]:
                flat_objects[rec]["children"].append(flat_objects.pop(obj))
        return list(flat_objects.values())

    #TODO load expert actions
    @classmethod
    @override
    def _convert_task_data_format(cls, task_cfg, raw_task_data) -> dict:
        question_type_id = task_cfg['question_type']
        task_cfg['task_description'] = raw_task_data['target_status'][question_type_id]['question']
        task_cfg['task_type'] = question_type_id
        task_cfg['target_status'] = raw_task_data['target_status'][question_type_id]

        scence_data = raw_task_data
        world_data = dict()

        # convert room
        rooms = []
        r_id = scence_data["sceneName"]
        geo_raw = scence_data["sceneBounds"]['cornerPoints']
        rooms.append({"id": r_id, "floorPolygon": geo_raw})
        world_data["rooms"] = rooms

        # convert object
        converted = WUtils.ai2thor_obj_2_procthor(scence_data["objects"])
        world_data["objects"] = converted[0]
        world_data["doors"] = converted[1]
        world_data["windows"] = converted[2]

        ag_init = scence_data["agent"]
        ag_rot = ag_init["rotation"]["y"]
        ag_pos = {"x": ag_init['position']["x"], "z": ag_init['position']["z"]}

        agents_data = {
            "position": ag_pos,
            "rotation": ag_rot
        }

        world_data["metadata"] = dict()
        world_data["metadata"]["agent"] = agents_data

        task_cfg["world_data"] = world_data
        task_cfg["world_type"] = cls.world_type

        with open(task_cfg["template"], "r") as template_file:
            template = template_utils.split(json.load(template_file))
            template["example"]["default"] = template["example"][str(question_type_id)]

        task_cfg["agents"][0]["template"] = template


        return task_cfg
