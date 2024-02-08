from __future__ import annotations

import re
from abc import ABC, abstractmethod


class OutputParser(ABC):
    @abstractmethod
    def parse(self, response):
        raise NotImplementedError


class RegexOutputParser(OutputParser):
    # action regex in the form of move_ahead|move_back|turn_left|turn_right|stop
    ALFRED_ACTION_REGEX = r"(\w+)\s*(?:\[(.*?)\])?"
    BABYAI_ACTION_REGEX = r"(move_ahead|turn_left|turn_right|stop|pick_up|drop|toggle)[\s\[]*(?:(red|green|blue|yellow|purple|grey)*[\s_]*(box|key|ball|door)?)?"
    ALFRED_ACTION_REGEX1 = r"(move_ahead|turn_left|turn_right|stop|pick_up|drop|toggle_on|toggle_off|put|slice|open|close|goto|heat|cool|clean|inspect|look)[\s\[]*(?:([a-zA-Z\|_]*_[0-9])*[\s,]*([a-zA-Z\|_]*_[0-9]*))?"
    REARRANGE_ACTION_REGEX = r"(move_ahead|turn_left|turn_right|stop|pick_up|pickup|drop|open|close|goto)[\s\[]*([a-zA-Z\|_]*_[0-9]*)?"

    def __init__(self, regex):
        self.regex = regex

    def parse(self, response: str):
        return re.findall(self.regex, response)
