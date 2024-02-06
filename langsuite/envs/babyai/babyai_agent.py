# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations
import json

import re
from copy import deepcopy
from math import floor
from typing import Dict

import numpy as np
import requests

from langsuite.actions import get_action
from langsuite.actions.base_action import BabyAIActionFeedback
from langsuite.agents.base_agent import AGENT_REGISTRY, Agent
from langsuite.envs.babyai.bot import BabyAIBot
from langsuite.llms import create_llm, create_llm_prompts, process_llm_results
from langsuite.llms.output_parsers import RegexOutputParser
from langsuite.shapes import Point2D, Vector2D
from langsuite.utils import logging
from langsuite.utils.logging import logger



def llm_gpt_35(messages, max_gen=100):
    # 替换为自己的KEY
    messages = [{"role": message["role"], "content": message["content"]} for message in messages]
    api_key = ""
    try:
        api_url = 'https://one.aiskt.com/v1/chat/completions'
        # 设置请求头部，包括 API 密钥
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        # 准备请求的数据
        payload = {
            'model': "gpt-3.5-turbo-16k",
            'messages': messages
        }
        # 发送 POST 请求
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
#        print(response.text)
        # 检查响应状态
        if response.status_code == 200:
            # 解析响应并提取需要的信息
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f'Error: Received status code {response.status_code}'
    except Exception as e:
        return 'An error occurred while sending the request'



@AGENT_REGISTRY.register()
class BabyAIAgent(Agent):
    def __init__(self, agent_config: Dict) -> None:
        super().__init__(agent_config=agent_config)

        self.llm = create_llm(agent_config.get("llm", None))

        self.set_config(agent_config)
        self.view_vector = None
        self.position = agent_config.get("position", (-1, -1))
        self.agent_dir = agent_config.get("agent_dir", -1)
        self.max_view_distance = agent_config.get("max_view_distance", 2)
        self._update()

        self.vis_mask = None
        self.inventory = []
        logger._logger.info(f"Successfully add agent: {self.cfg}")
        self.output_parser = RegexOutputParser(RegexOutputParser.BABYAI_ACTION_REGEX)
        self.status = dict(started=False)
        self.chat_history = []
        self.error_cache = []
        self.previous_action = None
        self.history_all = {}

    @classmethod
    def create(cls, agent_cfg: Dict):
        return cls(agent_config=agent_cfg)

    def _update(self):
        if self.agent_dir == 0:
            # right
            self.view_vector = Vector2D(1, 0)
        elif self.agent_dir == 1:
            # down
            self.view_vector = Vector2D(0, -1)
        elif self.agent_dir == 2:
            # left
            self.view_vector = Vector2D(-1, 0)
        elif self.agent_dir == 3:
            # up
            self.view_vector = Vector2D(0, 1)

    def step(self, action_dict):
        task_success = ""
        if type(action_dict) == str:
            action_dict = dict(prompt=action_dict)

        prompt = action_dict.get("prompt")
        if self.previous_action is None or self.previous_action["success"]:
            self.error_cache.clear()
            self.chat_history.append(
                {"role": "system", "content": action_dict["prompt"], "success": True}
            )
        else:
            self.error_cache.append(
                {"role": "system", "content": action_dict["prompt"], "success": False}
            )
        parsed_response = {}
        response = self.fetch_prompt_response(prompt)
        parsed_response = self.parse_response(response)
        logger._logger.info(parsed_response)
        success = parsed_response.get("success", True)
        if (
            success
            and "action" in parsed_response
            and parsed_response["action"] != "UserInput"
        ):
            if parsed_response["action"] == "BabyAIStop":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "BabyAIStop"
                )
                success = True
            else:
                action_status = self.execute(
                    action=parsed_response["action"],
                    **parsed_response["action_arg"],
                )
                # TODO what if action_status is None
                if action_status is None:
                    action_status = BabyAIActionFeedback(
                        success=False,
                        task_success=False,
                        feedback=self.env.feedback_builder.build(
                            'InvalidAction',
                            'failure.actionNotFound'
                        )
                    )
                logger._logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
                task_success = action_status.task_success

        if success:
            self.chat_history.append(
                {"role": "assistant", "content": response, "success": success}
            )
        else:
            self.error_cache.append(
                {"role": "assistant", "content": response, "success": success}
            )
        self.previous_action = dict(
            action=parsed_response.get("action"),
            success=success,
        )
        if success and task_success:
            parsed_response.update({"is_successfull": True})
            self.previous_action.update({"task_success": True})
        return success, parsed_response

    def execute(self, *, action: str = None, **action_kwargs):
        logger._logger.info(f"Working on action {action}")
        if not self.is_valid_action(action):
            logger._logger.info(f"Invalid action: {action}")
            return BabyAIActionFeedback(
                success=False, task_success=False, feedback=f"Invalid action: {action}"
            )

        action_or = get_action(
            action_name=action,
            env=self.env,
            agent=self,
        )
        if action_or:
            return action_or.step(**action_kwargs)
        else:
            logger._logger.info(f"Action {action} not found in environment.")
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=f"Action {action} not found in environment.",
            )

    def _justify_action_name(self, name):
        name = name.lower()
        splits = name.split("_")
        return "BabyAI" + "".join([s[0].upper() + s[1:] for s in splits])

    def _get_obj_type_and_color(self, obj_name):
        obj_patern = re.compile(r"box|key|ball|door")
        color_patern = re.compile(r"red|green|blue|yellow|purple|grey")
        object_type = None
        color = None
        find_obj = obj_patern.search(obj_name)
        if find_obj:
            object_type = find_obj.group(0)
        find_color = color_patern.search(obj_name)
        if find_color:
            color = find_color.group(0)
        return object_type, color

    def parse_response(self, response):
        """
        if there exists an action in the response, then excecute response and return action feedback
        """
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.env.world.agent_view_size,
                        side_steps=floor(self.env.world.agent_view_size / 2),
                        example=self.env.feedback_builder.build("example"),
                    ),
                )
            else:
                self.status["started"] = True
                return dict(
                    success=True,
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "BabyAIStart",
                        task=self.env.get_task_def(),
                        observation=self.env.get_observation(self),
                    ),
                )
        else:
            parsed_actions = self.output_parser.parse(response.lower())
            action_dicts = []
            for action_name, color, obj_type in parsed_actions:
                if action_name in ["move_ahead", "turn_left", "turn_right", "stop"]:
                    action_dicts.append(
                        dict(action_name=self._justify_action_name(action_name))
                    )
                elif action_name in [
                    "pick_up",
                    "drop",
                    "toggle",
                ]:
                    # obj_name = action_arg.strip("'").strip('"')
                    # obj_type, color = self._get_obj_type_and_color(obj_name)
                    action_arg = {}
                    if obj_type != "":
                        action_arg["object_type"] = obj_type
                    if color != "":
                        action_arg["color"] = color
                    action_dicts.append(
                        dict(
                            action_name=self._justify_action_name(action_name),
                            action_arg=action_arg,
                        )
                    )

            if len(action_dicts) > 1:
                return dict(
                    success=False,
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "InvalidAction", "failure.multipleActions"
                    ),
                )
            elif len(action_dicts) == 0:
                return dict(
                    success=False,
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "InvalidAction", "failure.actionNotFound"
                    ),
                )
            return dict(
                response=response,
                action=action_dicts[0]["action_name"],
                action_arg=action_dicts[0].get("action_arg", {}),
            )

    def fetch_prompt_response(self, prompt, history=[]):
        prompts = deepcopy(self.chat_history)
        if len(self.error_cache) > 0:
            prompts += deepcopy(self.error_cache)

        self.history_all[f"{len(self.history_all)}"] = prompts[1:]

#        response = self.llm(messages=create_llm_prompts(messages=prompts))
        response = llm_gpt_35(prompts)

        logger._logger.info(response)
        return process_llm_results(response)

    def reset(self):
        self.env.world.reset(seed=self.env.world.seed)
        self.vis_mask = np.zeros(
            shape=(self.env.world.width, self.env.world.height), dtype=bool
        )

        self.inventory.clear()
        self.set_config(self.init_cfg)
        self.view_size = self.env.world.agent_view_size
        self.position = self.env.world.agent_pos

    def set_config(self, agent_config):
        self.cfg.update(agent_config)

    def set_name(self, name):
        self.name = name

    def can_manipulate(self, position: Point2D, manipulation_distance=None):
        pass

    def can_observe(self, geometry):
        pass

    def set_position(self, position: Point2D):
        self.position = position

    def rotate(self, angle):
        pass

    def get_rotation(self):
        pass

    def add_inventory(self, inventory) -> bool:
        """
        Returns:
            Success or not.
        """
        if len(self.inventory) < self.inventory_capacity:
            self.inventory.append(inventory)
            return True
        return False

    def plot(self, axes=None):
        pass

    def render(self, fig=None):
        pass

    def get_agent_location(self):
        pass


@AGENT_REGISTRY.register()
class BabyAIAgentZeroShot(BabyAIAgent):
    def step(self, action_dict):
        if type(action_dict) == str:
            action_dict = dict(prompt=action_dict)

        prompt = action_dict.get("prompt")
        parsed_response = {}
        response = self.fetch_prompt_response(prompt, self.chat_history)
        if not response:
            return False, dict(feedback=prompt)
        parsed_response = self.parse_response(response)
        logger._logger.info(parsed_response)
        success = True
        if "action" in parsed_response and parsed_response["action"] != "UserInput":
            if parsed_response["action"] == "BabyAIPass":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "BabyAIPass"
                )
                success = False
            else:
                action_status = self.execute(action=parsed_response["action"])
                logger._logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
        if success:
            self.chat_history += [
                {"role": "system", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        return success, parsed_response

    def parse_response(self, response):
        """
        if there exists an action in the response, then excecute response and return action feedback
        """
        if not self.status["started"]:
            self.status["started"] = True
            return dict(
                response=response,
                feedback=self.env.feedback_builder.build(
                    "BabyAIStart",
                    task=self.env.get_task_def(),
                    observation=self.env.get_observation(self.agent),
                ),
            )
        else:
            thought = []
            act = []
            chat = []
            for line in response.split("\n"):
                if any(
                    [
                        think_tok in line.lower()
                        for think_tok in ["thought:", "think:", "i think"]
                    ]
                ):
                    thought.append("".join(line.split(":")[1:]))
                elif any([act_tok in line.lower() for act_tok in ["act:", "action:"]]):
                    act.append("".join(line.split(":")[1:]))
                elif len(act) == 0 and len(thought) > 0:
                    thought.append(line)
                elif len(chat) == 0 and len(act) > 0:
                    act.append(line)
                else:
                    chat.append(line)
            action = None
            if len(act) > 0:
                act = " ".join(act).strip()
                if "move_ahead" in act.lower():
                    action = "BabyAIMoveAhead"
                elif "rotate_left" in act.lower():
                    action = "BabyAITurnLeft"
                elif "rotate_right" in act.lower():
                    action = "BabyAITurnRight"
                elif "pick" in act.lower():
                    action = "BabyAIPick"
                elif "drop" in act.lower():
                    action = "BabyAIDrop"
                elif "toggle" in act.lower():
                    action = "BabyAIToggle"
                elif "stop" in act.lower():
                    action = "BabyAIStop"
                else:
                    action = "BabyAIPass"
            response = []
            if len(chat) > 0:
                response.append(("chat", " ".join(chat)))
            if len(thought) > 0:
                response.append(("thought", " ".join(thought)))
            if len(act) > 0:
                response.append(("act", act))
                return dict(response=response, action=action)

            return dict(response=response, feedback="OK.")


@AGENT_REGISTRY.register()
class BabyAIAgentReact(BabyAIAgent):
    def parse_response(self, response):
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.env.world.agent_view_size,
                        side_steps=floor(self.env.world.agent_view_size / 2),
                        example=self.env.feedback_builder.build("example"),
                    ),
                )
            else:
                self.status["started"] = True
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "BabyAIStart",
                        task=self.env.get_task_def(),
                        observation=self.env.get_observation(self),
                    ),
                )
        else:
            thought = []
            act = []
            for line in response.split("\n"):
                line = line.strip()
                if len(line) == 0:
                    continue
                if any(
                    [
                        think_tok in line.lower()
                        for think_tok in ["thought:", "think:", "i think:"]
                    ]
                ):
                    thought.append("".join(line.split(":")[1:]))
                elif any([act_tok in line.lower() for act_tok in ["act:", "action:"]]):
                    act.append("".join(line.split(":")[1:]))
                elif len(act) == 0 and len(thought) > 0:
                    thought.append(line)
                else:
                    act.append(line)
            response = []
            if len(thought) > 0:
                response.append(("thought", " ".join(thought)))
            if len(act) > 0:
                act = " ".join(act).strip()
                act_resp = super().parse_response(act)
                response.append(("act", act))
                return dict(
                    success=act_resp.get("success", True),
                    response=response,
                    action=act_resp.get("action", "Pass"),
                    action_arg=act_resp.get("action_arg", {}),
                    feedback=act_resp.get("feedback", None),
                )
            return dict(response=response, feedback="OK.", success=True)

@AGENT_REGISTRY.register()
class BabyAIAgentReflexion(BabyAIAgent):
    def __init__(self, agent_config: Dict) -> None:
        super().__init__(agent_config)
        self.history = dict()
        with open('/home/wtding/langsuite-dev/babyai_gpt3.5_reflexion_emmem/save/memory.txt', 'r') as log_file:
            for line in log_file.readlines():
                data: dict = json.loads(line.strip())
                for k, v in data.items():
                    self.history[k] = v

    def parse_response(self, response):
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.env.world.agent_view_size,
                        side_steps=floor(self.env.world.agent_view_size / 2),
                        example=self.env.feedback_builder.build("example"),
                    ),
                )
            else:
                self.status["started"] = True
                history = self.history.get(self.env.task_log_file_name)
                logging.logger._logger.info(f'{self.env.task_log_file_name}: {history}')
                # if history is not None:
                #     desc = self.env.get_task_def() + 'Your memory for the task is below:\n' + history
                # else:
                #     desc = self.env.get_task_def()
                if history is None or self.env.get_task_def() is None:
                    info = f"Do not need refleaction for {self.env.task_log_file_name}"
                    raise Exception(info)
                desc = self.env.get_task_def() + 'Your memory for the task is below:\n' + history
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "BabyAIStart",
                        task=desc,
                        observation=self.env.get_observation(self),
                    ),
                )
        else:
            thought = []
            act = []
            for line in response.split("\n"):
                line = line.strip()
                if len(line) == 0:
                    continue
                if any(
                    [
                        think_tok in line.lower()
                        for think_tok in ["thought:", "think:", "i think:"]
                    ]
                ):
                    thought.append("".join(line.split(":")[1:]))
                elif any([act_tok in line.lower() for act_tok in ["act:", "action:"]]):
                    act.append("".join(line.split(":")[1:]))
                elif len(act) == 0 and len(thought) > 0:
                    thought.append(line)
                else:
                    act.append(line)
            response = []
            if len(thought) > 0:
                response.append(("thought", " ".join(thought)))
            if len(act) > 0:
                act = " ".join(act).strip()
                act_resp = super().parse_response(act)
                response.append(("act", act))
                return dict(
                    success=act_resp.get("success", True),
                    response=response,
                    action=act_resp.get("action", "Pass"),
                    action_arg=act_resp.get("action_arg", {}),
                    feedback=act_resp.get("feedback", None),
                )
            return dict(response=response, feedback="OK.", success=True)

@AGENT_REGISTRY.register()
class BabyAIExpertAgent(BabyAIAgent):
    def __init__(self, agent_config: Dict) -> None:
        super().__init__(agent_config=agent_config)

        self.expert = None
        self.actions = None

    @classmethod
    def create(cls, agent_cfg: Dict):
        return cls(agent_config=agent_cfg)

    def generate_plan(self):
        self.reset()
        self.expert = BabyAIBot(self.env.world)
        action_map = {
            "forward": "move_ahead",
            "drop": "drop",
            "left": "turn_left",
            "right": "turn_right",
            "toggle": "toggle",
            "pickup": "pick_up",
        }

        action = self.expert.replan()
        actions = [
            {
                "action": "start",
                "agent_pos": self.expert.mission.agent_pos,
                "agent_dir": self.expert.mission.agent_dir,
            }
        ]
        while True:
            if action == self.expert.mission.actions.drop:
                actions.append(
                    {
                        "action": "drop",
                        "type": self.expert.mission.carrying.type,
                        "color": self.expert.mission.carrying.color,
                        "agent_pos": self.expert.mission.agent_pos,
                        "agent_dir": self.expert.mission.agent_dir,
                    }
                )
            elif action == self.expert.mission.actions.done:
                fwd_pos = self.expert.mission.front_pos
                fwd_cell = self.expert.mission.grid.get(*fwd_pos)
                actions.append(
                    {
                        "action": "stop []",
                        "agent_pos": self.expert.mission.agent_pos,
                        "agent_dir": self.expert.mission.agent_dir,
                    }
                )
                break

            else:
                action_t = action_map[str(action).split(".")[1]]
                if action_t in ["pick_up", "toggle"]:
                    fwd_pos = self.expert.mission.front_pos
                    fwd_cell = self.expert.mission.grid.get(*fwd_pos)
                    actions.append(
                        {
                            "action": action_t,
                            "type": fwd_cell.type,
                            "color": fwd_cell.color,
                            "agent_pos": self.expert.mission.agent_pos,
                            "agent_dir": self.expert.mission.agent_dir,
                        }
                    )
                else:
                    actions.append(
                        {
                            "action": action_t,
                            "agent_pos": self.expert.mission.agent_pos,
                            "agent_dir": self.expert.mission.agent_dir,
                        }
                    )
            _, _, done, _, _ = self.expert.mission.step(action)
            if done:
                actions.append(
                    {
                        "action": "stop",
                        "agent_pos": self.expert.mission.agent_pos,
                        "agent_dir": self.expert.mission.agent_dir,
                    }
                )
                break
            action = self.expert.replan()
        self.reset()
        return actions

    def step(self, action_dict):
        if self.expert is None:
            self.actions = self.generate_plan()

        parsed_response = {}
        task_success = ""
        if len(self.actions) == 0:
            parsed_response["response"] = [("act", "stop []")]
            parsed_response["action"] = "BabyAIStop"
            parsed_response["feedback"] = self.env.feedback_builder.build(
                "BabyAIStop", "default"
            )
            success = True
            return success, parsed_response
        expert_action = self.actions.pop(0)
        parsed_response = self.parse_expert_action(expert_action)
        logger._logger.info(parsed_response)
        success = True
        if "action" in parsed_response and parsed_response["action"] != "UserInput":
            if parsed_response["action"] == "BabyAIPass":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "BabyAIPass"
                )
                success = False
            elif parsed_response["action"] == "BabyAIStop":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "BabyAIStop", "default"
                )
                success = True
            else:
                # cur_pos_ = self.env.world.agent_pos
                # cur_dir = self.env.world.agent_dir
                action_status = self.execute(
                    action=parsed_response["action"], **parsed_response["action_arg"]
                )
                logger._logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
                task_success = action_status.task_success

            if success and task_success:
                parsed_response.update({"is_successfull": True})
        return success, parsed_response

    def execute(self, *, action: str = None, **action_kwargs):
        logger._logger.info(f"Working on action {action}")
        if not self.is_valid_action(action):
            logger._logger.info(f"Invalid action: {action}")
            return BabyAIActionFeedback(
                success=False, task_success=False, feedback=f"Invalid action: {action}"
            )

        action_or = get_action(
            action_name=action,
            env=self.env,
            agent=self,
        )
        if action_or:
            return action_or.step(**action_kwargs)
        else:
            logger._logger.info(f"Action {action} not found in environment.")
            return BabyAIActionFeedback(
                success=False,
                task_success=False,
                feedback=f"Action {action} not found in environment.",
            )

    def parse_expert_action(self, expert_action):
        if not self.status["started"]:
            self.status["started"] = True
            return dict(
                success=True,
                response="YES.",
                feedback=self.env.feedback_builder.build(
                    "BabyAIStart",
                    task=self.env.get_task_def(),
                    observation=self.env.get_observation(self),
                ),
            )
        else:
            color = None
            object_type = None
            action = expert_action.get("action")
            if "type" in expert_action:
                object_type = expert_action.get("type")
            if "color" in expert_action:
                color = expert_action.get("color")

            action_arg = {}
            if expert_action.get("action") == "move_ahead":
                action = "BabyAIMoveAhead"
            elif expert_action.get("action") == "turn_left":
                action = "BabyAITurnLeft"
            elif expert_action.get("action") == "turn_right":
                action = "BabyAITurnRight"
            elif expert_action.get("action") == "pick_up":
                action = "BabyAIPickUp"
            elif expert_action.get("action") == "drop":
                action = "BabyAIDrop"
            elif expert_action.get("action") == "toggle":
                action = "BabyAIToggle"
            elif expert_action.get("action") == "stop":
                action = "BabyAIStop"

            else:
                action = "BabyAIPass"

            act_str = expert_action.get("action")
            act_arg_str = []
            if color:
                action_arg["color"] = color
                act_arg_str.append(color)
            if object_type:
                action_arg["object_type"] = object_type
                act_arg_str.append(object_type)

            response = []
            if len(act_arg_str) > 0:
                response.append(("act", act_str + " [" + " ".join(act_arg_str) + "]"))

            elif act_str == "stop":
                response.append(("act", act_str + " []"))
            else:
                response.append(("act", act_str))
            return dict(
                response=response,
                action=action,
                action_arg=action_arg,
            )
