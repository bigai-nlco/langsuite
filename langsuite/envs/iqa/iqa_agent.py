# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import re
from copy import deepcopy

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.agents.simple_agent import SimpleAgent
from langsuite.llms import create_llm, create_llm_prompts, process_llm_results
from langsuite.utils.logging import logger


def parse_answer(response):
    pattern = r"Robot\.Answer\((.*?)\)"
    matches = re.findall(pattern, response)

    for match in matches:
        return match
    return ""


def parse_method(response, method):
    method = method.lower()
    pattern = r"" + method + "\s*\[([^]]+)\]"
    matches = re.findall(pattern.lower(), response.lower())

    for match in matches:
        return match
    return ""


@AGENT_REGISTRY.register()
class IqaAgent(SimpleAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config)

        self.chat_history = []
        self.status = dict(started=False)
        self.chat_history = []
        self.max_view_distance = None
        self.max_manipulate_distance = None
        self.view_degree = None
        self.llm = create_llm(agent_config["llm"])

    def step(self, action_dict):
        if type(action_dict) == str:
            action_dict = dict(prompt=action_dict)
        prompt = action_dict.get("prompt")
        if prompt:
            response = self.fetch_prompt_response(prompt)
            content = response
            parsed_response = self.parse_response(content)
            success = True
            if (
                "action" in parsed_response
                and parsed_response["action"].lower() == "answer"
            ):
                return success, parsed_response
            elif (
                "action" in parsed_response
                and parsed_response["action"].lower() == "open"
            ):
                object_id = parse_method(parsed_response["response"], "Open")
                object_name = ""
                try:
                    object_name = self.env.object_name2id[object_id]
                except:
                    return False, {}
                action_status = self._mock_execute(
                    action=parsed_response["action"], object_id=object_name
                )
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
            elif (
                "action" in parsed_response and parsed_response["action"] != "UserInput"
            ):
                action_status = self._mock_execute(action=parsed_response["action"])
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
            return success, parsed_response
        return False, {}

    def parse_response(self, response):
        """
        if there exists an action in the response, then excecute response and return action feedback
        """
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response, feedback=self.env.feedback_builder.build("intro")
                )
            else:
                self.status["started"] = True
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "IqaStart",
                        task=self.env.get_task_def(),
                        object_str=self.env.get_observation(self),
                    ),
                )
        elif "answer" in response.lower():
            response_answer = parse_answer(response)
            gold_answer = self.env.get_answer()
            return dict(
                response=response,
                action="Answer",
                answer_info="You are Right!"
                if response_answer == gold_answer
                else "You are Wrong!",
            )

        else:
            if "move_ahead" in response:
                action = "MoveAhead"
                response = "move_ahead"
            elif "turn_left" in response:
                action = "TurnLeft"
                response = "turn_left"
            elif "turn_right" in response:
                action = "TurnRight"
                response = "turn_right"
            elif "open" in response.lower():
                action = "Open"
                object_name = parse_method(response, "Open")
                if not object_name or object_name == "object_name":
                    return dict(response=response, action="pass")
                response = "open[{}]".format(object_name)
                return dict(
                    response=response,
                    action=action,
                    action_arg={"object_name": object_name},
                )
            elif "Stop" in response:
                action = "Stop"
            else:
                action = "Pass"
            return dict(response=response, action=action)

    def fetch_prompt_response(self, prompt):
        prompts = deepcopy(self.chat_history)
        prompts.append({"role": "system", "content": str(prompt)})
        self.chat_history.append({"role": "system", "content": str(prompt)})
        response = self.llm(messages=create_llm_prompts(messages=prompts))
        logger.info(response)

        return process_llm_results(response)

    def _mock_execute(self, *, action: str = None, **action_kwargs):
        logger.info(f"Working on action {action}")
        if not self.is_valid_action(action):
            logger.info(f"Invalid action: {action}")
            return ActionFeedback(success=False, feedback=f"Invalid action: {action}")

        action_or = get_action(action_name=action, env=self.env, agent=self)
        if action_or:
            return action_or.step(**action_kwargs)
        else:
            logger.info(f"Action {action} not found in environment.")
            return ActionFeedback(
                success=False, feedback=f"Action {action} not found in environment."
            )

    def set_name(self, name):
        self.name = name
