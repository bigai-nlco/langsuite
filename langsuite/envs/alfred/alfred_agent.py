# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
import json
import re
import time
from copy import deepcopy
from typing import Dict

import requests

import openai

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.agents.simple_agent import SimpleAgent
from langsuite.llms import create_llm, create_llm_prompts, process_llm_results
from langsuite.llms.output_parsers import RegexOutputParser
from langsuite.shapes import Vector2D
from langsuite.utils import math_utils
from langsuite.utils.logging import logger
from langsuite.utils.string_utils import camelcase
import datetime
# def llm_gpt_35(messages, max_gen=100):
#     for m in messages:
#         if "success" in m:
#             del m["success"]
#     rsp = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-16k",
#         messages=messages,
#         temperature=0.0,
#         top_p=1,
#         max_tokens=max_gen,
#         frequency_penalty=0,
#         presence_penalty=0,
#     )
#     response = (
#         rsp["choices"][0]["message"]["content"]
#         .replace("\n ", " ")
#         .strip(" ")
#         .strip('"')
#     )
#     return response

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
            'messages': messages,
            'temperature': 0
        }
        # 发送 POST 请求
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        # print(response)
        # 检查响应状态
        if response.status_code == 200:
            # 解析响应并提取需要的信息
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f'Error: Received status code {response.status_code}'
    except Exception as e:
        return 'An error occurred while sending the request'
    # count = 0
    # while True:
    #     if count > 3:
    #         break
    #     try:
    #         rsp = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo-16k",
    #             messages=messages,
    #             temperature=0.0,
    #             top_p=1,
    #             max_tokens=max_gen,
    #             frequency_penalty=0,
    #             presence_penalty=0,
    #         )
    #         print(rsp)
    #         response = (
    #             rsp["choices"][0]["message"]["content"]
    #             .replace("\n ", " ")
    #             .strip(" ")
    #             .strip('"')
    #         )
    #         return response
    #     except Exception as e:
    #         time.sleep(2)
    #         print(e)
    #         count += 1
    #         continue

    # return ""


@AGENT_REGISTRY.register()
class AlfredAgent(SimpleAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config=agent_config)
        self.status = dict(started=False)
        self.chat_history = []
        self.task_description = agent_config.get("task_description")

        # self.llm = create_llm(agent_config.get("llm"))
        self.output_parser = RegexOutputParser(RegexOutputParser.ALFRED_ACTION_REGEX1)
        self.error_cache = []
        self.previous_action = None
        self.history_all = {}

    def set_expert_actions(self, config):
        return []

    def step(self, action_dict):
        if type(action_dict) == str:
            action_dict = dict(prompt=action_dict)

        prompt = action_dict.get("prompt")
        action = action_dict.get("action")

        if prompt:
        
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
        if not action and prompt and len(prompt) > 0:
            response = self.fetch_prompt_response(prompt)
        else:
            response = action
        parsed_response = self.parse_response(response)
        logger.info(parsed_response)
        success = parsed_response.get("success", True)
        if (
            success
            and "action" in parsed_response
            and parsed_response["action"] != "UserInput"
        ):
            if parsed_response["action"] == "Stop":
                parsed_response["feedback"] = self.env.feedback_builder.build("Stop")
                success = True
            else:
                action_status = self.execute(
                    action=parsed_response["action"],
                    **parsed_response["action_arg"],
                )
                logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success

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
        return success, parsed_response

    def execute(self, *, action: str = None, **action_kwargs):
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

    def _get_obj_id_by_name(self, name):
        """Get object id by name"""
        objects_in_view = self.env.get_observed_objects(agent=self)
        # possible_objects = []
        objname_patern = re.compile(r"[a-zA-Z_]+_[0-9]+")
        match = objname_patern.match(name)
        if match and name in self.env.object_name2id:
            target_id = self.env.object_name2id[name]
            return target_id
            # if target_id in objects_in_view:
            #     return target_id
        else:
            possible_objects = []
            for id, o in objects_in_view.items():
                iter_type = camelcase(id.split("|")[0]).lower()
                if iter_type == name:
                    pose_info = o.get_obj_pose_info()
                    pose_info["distance"] = math_utils.euclidean_distance(
                        self.position, o.position
                    )
                    possible_objects.append(pose_info)
            if len(possible_objects) > 0:
                possible_objects = sorted(
                    possible_objects, key=lambda po: (po["distance"], po["name"])
                )
                return possible_objects[0]["objectId"]
        return None

    def get_object_in_inventory(self, object_id: str):
        for obj in self.inventory:
            if obj.id == object_id:
                return obj
        return None

    def _justify_action_name(self, name):
        name = name.lower()
        splits = name.split("_")
        return "".join([s[0].upper() + s[1:] for s in splits])

    def parse_response(self, response):
        """
        if there exists an action in the response, then excecute response and return action feedback
        """
        obj_id = None

        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.max_view_distance / self.step_size,
                        degree=self.aov / 2,
                        max_inventory=self.inventory_capacity,
                        max_manipulation_steps=self.max_manipulate_distance
                        / self.step_size,
                    ),
                )
            else:
                self.status["started"] = True
                return dict(
                    success=True,
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "Start",
                        task_description=self.task_description,
                        observation=self.env.get_look_around_observation(self),
                        # observation=self.env.get_observation(self),
                    ),
                )
        elif "feedback:" in response.lower() or "obs:" in response.lower():
            return dict(
                success=False,
                response=response,
                feedback=self.env.feedback_builder.build(
                    "InvalidAction", "failure.selfFeedback"
                ),
            )
        else:
            parsed_actions = self.output_parser.parse(response.lower())
            action_dicts = []
            for action_name, action_arg1, action_arg2 in parsed_actions:
                if action_name in ["move_ahead", "turn_left", "turn_right", "stop"]:
                    action_dicts.append(
                        dict(action_name=self._justify_action_name(action_name))
                    )
                elif action_name in [
                    "pick_up",
                    "drop",
                    "open",
                    "close",
                    "toggle_on",
                    "toggle_off",
                    "slice",
                    "goto",
                    "inspect"
                ]:
                    obj_name = action_arg2.strip("'").strip('"')
                    obj_id = self._get_obj_id_by_name(obj_name)
                    logger.info("obj name: {} obj id {}.".format(obj_name, obj_id))
                    if obj_id is not None:
                        action_dicts.append(
                            dict(
                                action_name=self._justify_action_name(action_name),
                                action_arg={
                                    "object_id": obj_id,
                                    "object_name": obj_name,
                                },
                            )
                        )
                    else:
                        action_dicts.append(
                            dict(
                                action_name=self._justify_action_name(action_name),
                            )
                        )
                elif action_name in [
                    "put",
                    "heat",
                    "clean",
                    "cool"
                ]:
                    obj_name = action_arg1.strip("'").strip('"')
                    obj_id = self._get_obj_id_by_name(obj_name)
                    receptacle_name = action_arg2.strip("'").strip('"')
                    receptacle_id = self._get_obj_id_by_name(receptacle_name)
                    logger.info("obj name: {} obj id {}.".format(obj_name, obj_id))
                    if obj_id is not None and receptacle_id is not None:
                        action_dicts.append(
                            dict(
                                action_name=self._justify_action_name(action_name),
                                action_arg={
                                    "object_id": obj_id,
                                    "object_name": obj_name,
                                    "receptacle_id": receptacle_id,
                                    "receptacle_name": receptacle_name,
                                },
                            )
                        )
                    else:
                        action_dicts.append(
                            dict(
                                action_name=self._justify_action_name(action_name),
                            )
                        )

            if len(action_dicts) > 1:
                return dict(
                    response=response,
                    action=action_dicts[0]["action_name"],
                    action_arg=action_dicts[0].get("action_arg", {}),
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

    def fetch_prompt_response(self, prompt):
        prompts = deepcopy(self.chat_history)
        if len(self.error_cache) > 0:
            prompts += deepcopy(self.error_cache)
        # prompts.append({"role": "system", "content": str(prompt)})
        logger.info(f"History: {prompts[1:]}")
        self.history_all[f"{len(self.history_all)}"] = prompts[1:]
        # with open("logs/history.json", "w") as fp:
        #     json.dump(self.history_all, fp, indent=4)
        self.chat_history.append({"role": "system", "content": str(prompt)})
        # response = self.llm(messages=create_llm_prompts(messages=prompts))
        # print("*****",prompt)
        # print(prompts)
        response = llm_gpt_35(prompts)
        logger.info(response)
        return process_llm_results(response)

    def get_inventory(self):
        if len(self.inventory) == 0:
            return "Empty"

        return ", ".join([self.env.object_id2name[o.id] for o in self.inventory])

    def reset(self):
        self.view_vector = Vector2D(0, 1)
        self.inventory.clear()
        self.set_config(self.init_cfg)

    def set_name(self, name):
        self.name = name


@AGENT_REGISTRY.register()
class AlfredAgentReact(AlfredAgent):
    def parse_response(self, response):
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.max_view_distance / self.step_size,
                        degree=self.aov / 2,
                        max_inventory=self.inventory_capacity,
                        max_manipulation_steps=self.max_manipulate_distance
                        / self.step_size,
                    ),
                )
            else:
                self.status["started"] = True
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "Start",
                        task_description=self.task_description,
                        observation=self.env.get_look_around_observation(self),
                        # observation=self.env.get_observation(self)
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
class AlfredExpertAgent(AlfredAgent):
    def __init__(self, agent_config: Dict) -> None:
        super().__init__(agent_config=agent_config)

        self.actions = None

    def set_expert_actions(self, expert_actions):
        actions = [{"action": "start"}]
        action_map = {
            "MoveAhead": "move_ahead",
            "Drop": "drop",
            "RotateLeft": "turn_left",
            "RotateRight": "turn_right",
            "ToggleObjectOn": "toggle_on",
            "ToggleObjectOff": "toggle_off",
            "PickupObject": "pick_up",
            "OpenObject": "open",
            "CloseObject": "close",
            "SliceObject": "slice",
            "PutObject": "put",
        }
        actions = [{"action": "start"}]
        for action in expert_actions:
            if "objectId" in action and action["action"] in action_map:
                if action["objectId"] in self.env.object_id2name:
                    obj_name = self.env.object_id2name[action["objectId"]]
                elif action["objectId"] == "DeskLamp|+03.14|+00.59|+01.47":
                    action["objectId"] = "DeskLamp|+03.14|+00.58|+01.47"
                    obj_name = self.env.object_id2name[action["objectId"]]

                else:
                    obj_name = (
                        action["objectId"].split("|")[0] + "|Sliced" + "_" + str(0)
                    )
                    obj_name = obj_name.lower()
                    self.env.object_id2name[action["objectId"]] = obj_name
                    self.env.object_name2id[obj_name] = action["objectId"]
                action_dic = {
                    "action": action_map[action["action"]],
                    "object": obj_name,
                }
                if (
                    "receptacleObjectId" in action
                    and action["action"] in action_map
                    and action["receptacleObjectId"] in self.env.object_id2name
                ):
                    action_dic["receptacle"] = self.env.object_id2name[
                        action["receptacleObjectId"]
                    ]
                actions.append(action_dic)
                # else:
                #     print(action["objectId"])
                #     actions = []
                #     break
            elif action["action"] in action_map:
                actions.append(
                    {
                        "action": action_map[action["action"]],
                    }
                )
            else:
                continue
        return actions

    def step(self, action_dict):
        parsed_response = {}
        if len(self.actions) == 0:
            parsed_response["response"] = [("act", "stop []")]
            parsed_response["action"] = "Stop"
            parsed_response["feedback"] = self.env.feedback_builder.build(
                "Stop", "default"
            )
            success = True
            return success, parsed_response
        expert_action = self.actions.pop(0)
        parsed_response = self.parse_expert_action(expert_action)
        logger.info(parsed_response)
        success = True
        if "action" in parsed_response and parsed_response["action"] != "UserInput":
            if parsed_response["action"] == "Pass":
                parsed_response["feedback"] = self.env.feedback_builder.build("Pass")
                success = False
            elif parsed_response["action"] == "Stop":
                parsed_response["feedback"] = self.env.feedback_builder.build(
                    "Stop", "success"
                )
                success = True
            else:
                # cur_pos_ = self.env.world.agent_pos
                # cur_dir = self.env.world.agent_dir
                start_time = datetime.datetime.now()
                action_status = self.execute(
                    action=parsed_response["action"], **parsed_response["action_arg"]
                )
                end_time = datetime.datetime.now()
                #print((end_time-start_time).microseconds)
                logger.info(action_status)
                parsed_response["feedback"] = action_status.feedback
                success = action_status.success
        return success, parsed_response

    def execute(self, *, action: str = None, **action_kwargs):
        logger.info(f"Working on action {action}")
        if not self.is_valid_action(action):
            logger.info(f"Invalid action: {action}")
            return ActionFeedback(success=False, feedback=f"Invalid action: {action}")

        action_or = get_action(
            action_name=action,
            env=self.env,
            agent=self,
        )
        if action_or:
            return action_or.step(**action_kwargs)
        else:
            logger.info(f"Action {action} not found in environment.")
            return ActionFeedback(
                success=False, feedback=f"Action {action} not found in environment."
            )

    def parse_expert_action(self, expert_action):
        if not self.status["started"]:
            self.status["started"] = True
            return dict(
                response=[("act", "YES.")],
                feedback=self.env.feedback_builder.build(
                    "Start",
                    task_description=self.env.get_task_def(),
                    observation=self.env.get_observation(self),
                ),
            )
        else:
            object_name = None
            receptacle_name = None
            if "object" in expert_action:
                object_name = expert_action.get("object")
            if "receptacle" in expert_action:
                receptacle_name = expert_action.get("receptacle")
            action_arg = {}
            act_str = expert_action.get("action")
            if act_str == "move_ahead":
                action = "MoveAhead"
            elif act_str == "turn_left":
                action = "TurnLeft"
            elif act_str == "turn_right":
                action = "TurnRight"
            elif act_str == "pick_up":
                action = "PickUp"
            elif act_str == "drop":
                action = "Drop"
            elif act_str == "toggle_on":
                action = "ToggleOn"
            elif act_str == "toggle_off":
                action = "ToggleOff"
            elif act_str == "slice":
                action = "Slice"
            elif act_str == "open":
                action = "Open"
            elif act_str == "close":
                action = "Close"
            elif act_str == "put":
                action = "Put"
            elif act_str == "stop":
                action = "Stop"
            else:
                action = "Pass"

            response = []
            if object_name is not None and receptacle_name is not None:
                action_arg["object_name"] = object_name
                action_arg["object_id"] = self.env.object_name2id[object_name]
                action_arg["receptacle_name"] = receptacle_name
                action_arg["receptacle_id"] = self.env.object_name2id[receptacle_name]
                response.append(
                    ("act", act_str + " [" + object_name + ", " + receptacle_name + "]")
                )
            elif object_name is not None:
                action_arg["object_name"] = object_name
                action_arg["object_id"] = self.env.object_name2id[object_name]
                response.append(("act", act_str + " [" + object_name + "]"))
            else:
                response.append(("act", act_str))
            return dict(
                response=response,
                action=action,
                action_arg=action_arg,
            )
