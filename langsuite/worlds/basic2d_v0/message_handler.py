from abc import abstractmethod
from email import message
import re
from typing import List, Optional, Tuple
from attr import dataclass
from git import Tree
from overrides import override
from langsuite.suit import InvalidActionError, ParameterMissingError
from langsuite.suit import MessageHandler
from langsuite.suit import Message
from langsuite.suit.exceptions import StructuredException
from langsuite.suit import TaskStatus
from langsuite.utils import logging

@dataclass
class Basic2DHandler(MessageHandler):
    task_type: str
    target_status: TaskStatus

    ACTIONS_WITH_NO_ARGS = {}
    ACTIONS_WITH_ONE_ARG = {}
    ACTIONS_WITH_TWO_ARGS = {}
    STOP_NAME = 'stop'

    @abstractmethod
    def mapping_action_names(self, s: str) -> str:
        pass

    def handle_actions(self, matched: re.Match):
        action_name = matched.group(1)
        action_arg1 = matched.group(2)
        action_arg2 = matched.group(3)
        logging.logger.debug(
        "action=%s, arg_1=%s, arg_2=%s",
            action_name, action_arg1, action_arg2
        )
        formal_name = self.mapping_action_names(action_name)
        action_dict: dict[str, object] = {"action": formal_name}
        try:
            if action_name in self.ACTIONS_WITH_NO_ARGS:
                assert (not action_arg1) and (not action_arg2)
            elif action_name in self.ACTIONS_WITH_ONE_ARG:
                assert action_arg1 and not action_arg2
            elif action_name in self.ACTIONS_WITH_TWO_ARGS:
                assert action_arg1 and action_arg2
        except AssertionError as e:
            raise ParameterMissingError(
                {"action": "InvalidAction", "status": "failure.actionNotFound"}
            ) from e
        if action_arg1:
            action_dict["object_index"] = action_arg1.strip("'").strip('"')
        if action_arg2:
            action_dict["receptacle_index"] = action_arg2.strip("'").strip('"')
        return action_dict

    @override
    def parse(self, agent_name, response) -> List[Tuple[Message, dict]]:
        if type(response) == list:
            result = []
            for res in response:
                result.extend(self.parse(agent_name, res))
            return result
            
        actions = "|".join(
            (
                "|".join(self.ACTIONS_WITH_NO_ARGS),
                "|".join(self.ACTIONS_WITH_ONE_ARG),
                "|".join(self.ACTIONS_WITH_TWO_ARGS),
            )
        )
        param_fmt = '([A-Za-z0-9_]+)'
        act_regex = fr"[Aa]ct(?:ion)?: *({actions}) *(?:\[{param_fmt}(?: *, *{param_fmt})?\])? *"
        stop_regex = fr"(?:[Aa]ct(?:ion)?:)? *{self.STOP_NAME} *\[(.+)\] *"
           
        act = []
        in_thought = False
        thought = ''
        for line in response['content'].split("\n"):
            #Should we support multi actions?
            if len(act) > 0:
                raise StructuredException({
                    "action": "InvalidAction",
                    "status": "failure.multipleActions"
                })
            matched = re.match(stop_regex, line)
            if matched:
                if in_thought:
                    in_thought = False
                    message = Message(
                        role='assistant',
                        raw_content=thought,
                        name=agent_name,
                        action='thought'
                    )
                    act.append((message, dict()))
                message = Message(
                    role='assistant',
                    raw_content=line,
                    name=agent_name,
                    action='act'
                )
                action = {       
                    'action': 'Stop',
                    'task_type': self.task_type,
                    'target_status': self.target_status,
                    'answer': matched.group(1)
                }
                act.append((message, action))
                continue
            matched = re.match(act_regex, line)
            if matched:
                if in_thought:
                    in_thought = False
                    message = Message(
                        role='assistant',
                        raw_content=thought,
                        name=agent_name,
                        action='thought'
                    )
                    act.append((message, dict()))

                parsed = self.handle_actions(matched)
                message = Message(
                    role='assistant',
                    raw_content=line,
                    name=agent_name,
                    action='act'
                )
                act.append((message, parsed))
                continue
            if line.startswith('think:') or line.startswith('thought:'):
                in_thought = True
            if in_thought:
                thought += line + '\n'          
        if in_thought:
            message = Message(
                role='assistant',
                raw_content=thought,
                name=agent_name,
                action='thought'
            )
            act.append((message, dict()))
        if len(act) == 0:
            raise StructuredException({
                    "action": "InvalidAction",
                    "status": "failure.actionNotFound"
                })

        return act
    
    @staticmethod
    def pack_desc(entity_info: dict) -> str:
        #        logging.logger.debug(entity_info)
        s = list()
        for key, value in entity_info.items():
            # TODO support configurations?
            if key.startswith("is") and key != 'isSliced' and not key.endswith("Source"):
                if value is True:
                    if key == 'isFilledWithLiquid':
                        s.append('filled')
                    else:
                        s.append(key[2:].lower())
                elif key == "isOpen" and value is False:
                    s.append("closed")
        if entity_info.get("temperature") in {"Hot", "Cold"}:
            s.append(entity_info["temperature"].lower())

        prefix = ", ".join(s)
        if len(prefix):
            prefix += " "

        if entity_info.get("isFilledWithLiquid") and not entity_info.get("isFilledWithLiquid") is True:
            suffix = f' filled with {entity_info["isFilledWithLiquid"]}'
        else:
            suffix = ""
        # XXX why use 'a', not 'the'? should I consider 'an' for aeiou?
        s = f'a {prefix}{entity_info["index"]}{suffix}'
        return s

    @staticmethod
    def pack_list_with_inner(list_info, described_collector, prefix) -> str:
        list_desc, desc_list = zip(
            *(
                Basic2DHandler.iter_pack(entity_info, described_collector)
                for entity_info in list_info
            )
        )
        list_desc = ", ".join(list_desc)
        desc_list: List[str] = list(filter(lambda x: isinstance(x, str), desc_list))
        desc_list.insert(0, f"{prefix}{list_desc}")
        return "; ".join(desc_list)

    @staticmethod
    def iter_pack(
        entity_info: dict, described_collector: set
    ) -> Tuple[str, Optional[str]]:
        name = entity_info["index"]
        if name in described_collector:
            return "", None
        described_collector.add(name)
        if entity_info.get("isOpen") is not False:  # True or None
            self_desc = Basic2DHandler.pack_desc(entity_info)
            if "content" not in entity_info:  # not a receptacle
                return f"{self_desc}", None
            if len(entity_info["content"]) == 0:
                suffix = ", it's empty" if entity_info.get("isOpen") is False else ""
                return f"{self_desc}{suffix}", None
            prefix = "In" if entity_info.get("isOpen") else "On"
            prefix = f"{prefix} {name}, you see "
            desc_list = Basic2DHandler.pack_list_with_inner(
                entity_info["content"], described_collector, prefix
            )
            return self_desc, desc_list
        else:
            return Basic2DHandler.pack_desc(entity_info), None

    @staticmethod
    def pack_list(objs, direction, described_collector) -> str:
        if len(objs) == 0:
            return ""
        prefix = f"{direction} you see "
        observation = Basic2DHandler.pack_list_with_inner(
            objs, described_collector, prefix
        )
        if len(observation) > 0:
            observation += ". "
        return observation

    @staticmethod
    @override
    def pack(semantic_observation: dict) -> str:
        middle_objs = semantic_observation["middle_objs"]
        left_objs = semantic_observation["left_objs"]
        right_objs = semantic_observation["right_objs"]

        described_collector = set()

        middle_observation = Basic2DHandler.pack_list(
            middle_objs, "In front of you,", described_collector
        )
        left_observation = Basic2DHandler.pack_list(
            left_objs, "On your left,", described_collector
        )
        right_observation = Basic2DHandler.pack_list(
            right_objs, "On your right,", described_collector
        )
        surface_observation = middle_observation + left_observation + right_observation
        if len(surface_observation) == 0:
            surface_observation = "You see nothing. You can try to take action like move_ahead, turn_left or turn_right to explore the room."

        return surface_observation
