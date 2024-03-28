# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-

import os
from copy import deepcopy

from langsuite.actions import ActionFeedback
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.agents.simple_agent import SimpleAgent

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def mock_openai_response(
    model="gpt-3.5-turbo",
    messages=[],
    temperature: float = 1.0,
    top_p: int = 1,
    stream: bool = False,
):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Ok, I can help you with this.",
                    "role": "assistant",
                },
            }
        ],
        "created": 123456,
        "model": model,
        "object": "chat.completion",
    }


@AGENT_REGISTRY.register()
class ChatGPTAgent(SimpleAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config)

        self.chat_history = []

    def step(self, action_dict):
        if type(action_dict) == str:
            action_dict = dict(prompt=action_dict)

        prompt = action_dict.get("prompt")
        if prompt:
            response = self.fetch_prompt_response(prompt)
            parsed_response = self.parse_response(response)
            success = True
            if "action" in parsed_response and parsed_response["action"] != "UserInput":
                action_status = self._mock_execute(parsed_response["action"])
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
            return success, parsed_response
        return False, {}

    def parse_response(self, response):
        """
        if there exists an action in the response, then excecute response and return action feedback
        """
        return dict(response=response, action="MockAction")

    def fetch_prompt_response(self, prompt, history=[]):
        prompts = deepcopy(history)
        prompts.append({"role": "system", "content": prompt})

        response = mock_openai_response(prompts)

        return response["choices"][0]["message"]["content"]

    def _mock_execute(self, action):
        return ActionFeedback(success=True, feedback="Action recevied")
