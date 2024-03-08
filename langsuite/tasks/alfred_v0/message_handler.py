from typing import ClassVar, Dict, List, Set, Tuple
from attr import dataclass
from overrides import override
from langsuite.suit import TaskStatus
from langsuite.suit.message import Message
from langsuite.worlds.basic2d_v0.message_handler import Basic2DHandler

@dataclass
class AlfredStatus(TaskStatus):
    mrecep_target: str
    object_sliced: bool
    object_target: str
    parent_target: str
    toggle_target: str

@dataclass
class AlfredHandler(Basic2DHandler):
    action_name_map: Dict[str, str]
    ACTIONS_WITH_NO_ARGS: Set[str] = {"move_ahead", "turn_left", "turn_right", "stop"}
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
        if 'objectId' in action['api_action']:
            object_id = action['api_action']['objectId']
            action_dict['object_id'] = object_id
            if 'receptacleObjectId' in action['api_action']:
                receptacle_id = action['api_action']['receptacleObjectId']
                action_dict["receptacle_id"] = receptacle_id
                args = f' [{object_id}, {receptacle_id}]'
            else:
                args = f' [{object_id}]'
            
        content = f'act: {action_name}{args}'
        message = Message(
            role='assistant',
            raw_content=content,
            name=agent_name,
            action='act'
        )
        return [(message, action_dict)]
