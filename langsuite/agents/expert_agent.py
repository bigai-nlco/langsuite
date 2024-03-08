from ast import parse
from copy import deepcopy
from email import message
import json
import re
from typing import List, Optional

from overrides import override
import requests
from langsuite.suit import AGENT_REGISTRY
from langsuite.suit import LangSuiteAgent
from langsuite.suit import Message
from langsuite.suit.exceptions import StructuredException
from langsuite.suit.message import MessageHandler
from langsuite.utils import logging


@AGENT_REGISTRY.register()
class ExpertAgent(LangSuiteAgent):
    def __init__(self, name, world, agent_data, step_limit):
        super().__init__(name, world, agent_data, 100000000)
        self.action_list: list = agent_data["expert_actions"]

    # TODO move to utils?
    def split(self, data):
        def split_t(template):
            params = re.findall(r"\{([^}]+)\}", template[0])
            return {"template": template[0], "params": params}

        return {
            act_name: {
                status: split_t(inner_value)
                for status, inner_value in inner_dict.items()
            }
            for act_name, inner_dict in data.items()
        }

    def pack_observation(self, sem_obs):
        if isinstance(sem_obs, str):
            # HACK Already packed
            return sem_obs
        else:
            return self.message_handler.pack(sem_obs)

    @override
    def setup(self, agent_config, cmd_cli, web_ui):
        super().setup(agent_config, cmd_cli, web_ui)

        with open(agent_config["template"], "r") as template_file:
            self.template = self.split(json.load(template_file))

        self.status = dict(started=False)
        self.chat_history = []

        # self.llm = create_llm(agent_config.get("llm"))
        self.message_handler: MessageHandler = agent_config.get("parser")
        self.error_cache = []
        self.previous_action = None
        self.history_all = {}

    @override
    def init(self, task_description: str) -> str:
        start_info = self.template["intro"]["default"]["template"]
        params = self.template["intro"]["default"]["params"]
        for param in params:
            if param == "example":
                start_info = start_info.replace(
                    "{" + param + "}", self.template["example"]["default"]["template"]
                )
            else:
                start_info = start_info.replace(
                    "{" + param + "}", str(self.body.__dict__[param])
                )
        logging.logger.debug(start_info)

        if self.cmd_cli:
            message = Message(role="system", raw_content=str(start_info), name=self.name)
            self.cmd_cli.agent_step(message)
            message = Message(role="assistant", raw_content=str("YES"), name=self.name)
            self.cmd_cli.agent_step(message)

        semantic_fb_obs = (
            {
                "task_description": task_description,
                "action": "Start",
            },
            self.world.get_observation(self.name),
        )
        logging.logger.debug("obs=%s", self.pack(semantic_fb_obs))
        return self.pack(semantic_fb_obs)

    @override
    def pack(
        self,
        semantic_fb_obs,
        template: Optional[str] = None,
        param: Optional[dict] = None,
    ) -> str:
        sem_feedback = semantic_fb_obs[0]
        sem_obs = semantic_fb_obs[1]
        action_name = sem_feedback["action"]
        # Is it good to use default or I should raise exceptions?
        status = sem_feedback.get("status", "default")
        if (template is None) or (param is None):
            try:
                logging.logger.debug("action: %s with status: %s", action_name, status)
                template = self.template[action_name][status]["template"]
                params = self.template[action_name][status]["params"]
            except KeyError:
                logging.logger.debug(
                    "status %s not in template, change to failure.default", status
                )
                template = self.template[action_name]["failure.default"]["template"]
                params = self.template[action_name]["failure.default"]["params"]
        for key in params:  # type: ignore
            if key == "observation":
                template = template.replace("{observation}", self.pack_observation(sem_obs))  # type: ignore
            else:
                assert key in sem_feedback
                template = template.replace("{" + key + "}", str(sem_feedback[key]))  # type: ignore
        return template  # type: ignore

    @override
    def fetch_response(self, prompt, role="system") -> dict:
        message = Message(role=role, raw_content=str(prompt), name=self.name)
        if self.cmd_cli:
            self.cmd_cli.agent_step(message)
        if self._stopped:
            return {}
        parsed = self.message_handler.parse_expert_action(self.name, self.action_list.pop(0))
        if self.cmd_cli:
            for message, _ in parsed:
                self.cmd_cli.agent_step(message)

        (message, action) = next(
            (message, action) for (message, action) in parsed if message.action == "act"
        )

        return action
