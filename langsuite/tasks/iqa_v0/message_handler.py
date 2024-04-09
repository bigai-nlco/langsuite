from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from overrides import override
from langsuite.suit import TaskStatus
from langsuite.suit.message import Message
from langsuite.worlds.basic2d_v0.message_handler import Basic2DHandler


@dataclass
class IQAStatus(TaskStatus):
    answer: Union[int, bool]
    object_class: str
    recept: Optional[str]


class IQAHandler(Basic2DHandler):
    ACTIONS_WITH_NO_ARGS = {"move_ahead", "turn_left", "turn_right"}
    ACTIONS_WITH_ONE_ARG = {"open"}
    ACTIONS_WITH_TWO_ARGS = set()
    STOP_NAME = "answer"

    def __init__(
        self, task_type: str, target_status: TaskStatus, action_name_map: Dict[str, str]
    ) -> None:
        super().__init__(task_type, target_status)
        self.action_name_map = action_name_map

    def __post_init__(self):
        self.ACTIONS_WITH_NO_ARGS.intersection_update(self.action_name_map.keys())
        self.ACTIONS_WITH_ONE_ARG.intersection_update(self.action_name_map.keys())
        self.ACTIONS_WITH_TWO_ARGS.intersection_update(self.action_name_map.keys())

    @override
    def mapping_action_names(self, s: str) -> str:
        return self.action_name_map[s]

    @override
    def parse_expert_action(
        self, agent_name, action: Dict
    ) -> List[Tuple[Message, dict]]:
        raise NotImplementedError("There is no IQA expert now.")
