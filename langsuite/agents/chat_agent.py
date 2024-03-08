from ast import parse
from copy import deepcopy
from email import message
import json
from multiprocessing import process
import random
import re
from typing import Optional

from overrides import override
import requests
from langsuite.cli.cmd_cli import CMDClient
from langsuite.llms import (
    create_llm,
    create_llm_prompts,
    langchain,
    process_llm_responses,
)
from langsuite.suit import AGENT_REGISTRY
from langsuite.suit import LangSuiteAgent
from langsuite.suit import Message
from langsuite.suit.exceptions import StructuredException
from langsuite.suit.message import MessageHandler
from langsuite.utils import debug_utils
from langsuite.utils.logging import logger


@AGENT_REGISTRY.register()
class ChatAgent(LangSuiteAgent):
    def __init__(self, name, world, agent_data, step_limit = 5):
        super().__init__(name, world, agent_data, step_limit)
        self.template = agent_data["template"]
        self.from_user = agent_data.get("from_user", False)
        self.debug = agent_data.get("debug", False)
        if not self.from_user:
            if not self.debug:
                self.llm = create_llm(agent_data.get("llm"))
            if self.llm is None:
                logger.warn("Failed to create LLM, will use user_input.")
            else:
                logger.warn("Created LLM: %s", self.llm)
        else:
            self.llm = None
        self.cmd_cli: Optional[CMDClient] = None
        self.web_ui = None
        

    def pack_observation(self, sem_obs):
        if isinstance(sem_obs, str):
            # HACK Already packed
            return sem_obs
        else:
            return self.message_handler.pack(sem_obs)

    def _select(self, templates: list):
        return random.choice(templates)

    @override
    def setup(self, agent_config, cmd_cli, web_ui):
        super().setup(agent_config, cmd_cli, web_ui)
        self.status = dict(started=False)
        self.chat_history = []

        # self.llm = create_llm(agent_config.get("llm"))
        self.message_handler: MessageHandler = agent_config.get("parser")
        # self.error_cache = []
        # self.previous_action = None
        # self.history_all = {}

    @override
    def init(self, task_description: str) -> str:
        source = random.choice(self.template["intro"]["default"])
        start_info = source["template"]
        params = source["params"]
        for param in params:
            if param == "example":
                start_info = start_info.replace(
                    "{" + param + "}", random.choice(self.template["example"]["default"])["template"]
                )
            else:
                start_info = start_info.replace(
                    "{" + param + "}", str(self.body.__dict__[param])
                )
        logger.debug(start_info)
        try:
            response = self.fetch_response(prompt=start_info, role="system")
        except StructuredException as e:
            if e.param_dict["status"] == "failure.actionNotFound":
                pass
            else:
                raise e

        semantic_fb_obs = (
            {
                "task_description": task_description,
                "action": "Start",
            },
            self.world.get_observation(self.name),
        )
        logger.debug("obs=%s", self.pack(semantic_fb_obs))
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
                logger.debug("action: %s with status: %s", action_name, status)
                source = random.choice(self.template[action_name][status])
            except KeyError:
                logger.debug(
                    "status %s not in template, change to failure.default", status
                )
                source = random.choice(self.template[action_name]["failure.default"])
            template = source["template"]
            params = source["params"]
        for key in params:  # type: ignore
            if key == "observation":
                template = template.replace("{observation}", self.pack_observation(sem_obs))  # type: ignore
            else:
                assert key in sem_feedback
                template = template.replace("{" + key + "}", str(sem_feedback[key]))  # type: ignore
        return template  # type: ignore

    @override
    def fetch_response(self, prompt, role="system") -> dict:
        messages = deepcopy(self.chat_history)
        # if len(self.error_cache) > 0:
        #     prompts += deepcopy(self.error_cache)
        message = Message(role=role, raw_content=str(prompt), name='env')
        messages.append(message)
        # self.history_all[f"{len(self.history_all)}"] = prompts[1:]
        self.chat_history.append(message)

        if self._stopped:
            if self.cmd_cli:
                self.cmd_cli.agent_step(message)
            return {}

        if self.llm is None or self.from_user:
            if self.cmd_cli:
                content = self.cmd_cli.agent_step(message, user_input=True)
            else:
                content = input()
            response = {
                "role": "user",
                "content": content,
            }
        else:
            if self.cmd_cli:
                 self.cmd_cli.agent_step(message)
            
            if self.debug:
                # Naive requests
                response = debug_utils.manual_request(messages)
            else:               
                # langchain
                prompts = create_llm_prompts(messages)
                response = self.llm(prompts)
                response = process_llm_responses(response, self.name)
            
            for message in response:
                msg = Message(role=message['role'], raw_content=message['content'], name=self.name)
                self.chat_history.append(msg)
                if self.cmd_cli:
                    self.cmd_cli.agent_step(msg)

        parsed = self.message_handler.parse(self.name, response)

        try:
            (message, action) = next(
                (message, action) for (message, action) in parsed if message.action == "act"
            )
        except StopIteration:
            #Only thought
            return {}
        
        return action
