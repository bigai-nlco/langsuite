from typing import ClassVar, Dict, List, Set, Tuple
from attr import dataclass
from overrides import override
from langsuite.suit import TaskStatus, world
from langsuite.suit.message import Message
from langsuite.worlds.basic2d_v0.message_handler import Basic2DHandler
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0

@dataclass
class AlfredStatus(TaskStatus):
    mrecep_target: str
    object_sliced: bool
    object_target: str
    parent_target: str
    toggle_target: str

@dataclass
class AlfredHandler(Basic2DHandler):
    world: Basic2DWorld_V0
    action_name_map: Dict[str, str]
    ACTIONS_WITH_NO_ARGS: Set[str] = {"move_ahead", "turn_left", "turn_right"}
    ACTIONS_WITH_ONE_ARG: Set[str] = {
        "pick_up",
        "drop",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        "slice",
        "goto",
    }
    ACTIONS_WITH_TWO_ARGS: Set[str] = {"put", "cook", "heat", "cool", "clean"}

    def __post_init__(self):
        self.ACTIONS_WITH_NO_ARGS.intersection_update(self.action_name_map.keys())
        self.ACTIONS_WITH_ONE_ARG.intersection_update(self.action_name_map.keys())
        self.ACTIONS_WITH_TWO_ARGS.intersection_update(self.action_name_map.keys())

    @override
    def mapping_action_names(self, s: str) -> str:
        return self.action_name_map[s]

    @override
    def parse_expert_action(self, agent_name, action: dict) -> List[Tuple[Message, dict]]:
        action_id = action['api_action']['action']
        action_name = action['action_name']
        args = ''
        action_dict = {
            'action': action_id
        }
        if action_id == 'Stop':
            action_dict['task_type'] = self.task_type
            action_dict['target_status'] = self.target_status
            action_dict['answer'] = 'n/a'
            args = ' [n/a]'
        if 'objectId' in action['api_action']:
            object_id = action['api_action']['objectId']
            object_index = self.world.object_name2index(object_id)
            action_dict['object_index'] = object_index
            if 'receptacleObjectId' in action['api_action']:
                receptacle_id = action['api_action']['receptacleObjectId']
                receptacle_index = self.world.object_name2index(receptacle_id)
                action_dict["receptacle_index"] = receptacle_index
                args = f' [{object_index}, {receptacle_index}]'
            else:
                args = f' [{object_index}]'

        content = f'act: {action_name}{args}'
        
        message = Message(
            role='assistant',
            raw_content=content,
            name=agent_name,
            action='act'
        )
        return [(message, action_dict)]
