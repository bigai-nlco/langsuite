from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Message:
    role: str
    raw_content: str
    name: str
    to: Optional[str] = None
    action: str = "chat"
    
    def __post_init__(self):
        # HACK It will be better to check the prefix carefully.
        if (self.action == 'act' or 'thought') and (':' in self.raw_content):
            self.stripped_content = self.raw_content.split(":", 1)[1]
        else:
            self.stripped_content = self.raw_content
        
    @property
    def dump_dict(self):
        return {"role": self.role, "content": self.raw_content}


class MessageHandler(ABC):
    @abstractmethod
    def parse(self, agent_name, response) -> List[Tuple[Message, dict]]:
        pass

    @staticmethod
    @abstractmethod
    def pack(semantic_observation: dict) -> str:
        pass

    @abstractmethod
    def parse_expert_action(self, agent_name, action: dict) -> List[Tuple[Message, dict]]:
        pass