# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import re
from copy import deepcopy
from math import floor

from langsuite.actions import ActionFeedback, get_action
from langsuite.agents.base_agent import AGENT_REGISTRY
from langsuite.agents.simple_agent import SimpleAgent
from langsuite.llms import create_llm, create_llm_prompts, process_llm_results
from langsuite.llms.output_parsers import RegexOutputParser
from langsuite.shapes import Vector2D
from langsuite.utils import math_utils
from langsuite.utils.logging import logger
from langsuite.utils.string_utils import camelcase


@AGENT_REGISTRY.register()
class RearrangeAgent(SimpleAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config=agent_config)
        self.status = dict(started=False)
        self.chat_history = []
        self.target_status = agent_config.get("target_status")
        self.start_status = agent_config.get("start_status")
        self.llm = create_llm(agent_config["llm"])
        self.output_parser = RegexOutputParser(RegexOutputParser.REARRANGE_ACTION_REGEX)
        self.error_cache = []
        self.previous_action = None
        self.history_all = {}

    def step(self, action_dict):
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
            # if target_id in objects_in_view:
            return target_id
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
                        max_view_steps=str(self.max_view_distance / self.step_size),
                        degree=floor(self.aov / 2),
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
                        task_description=self.env.target_pose_description,
                        observation=self.env.get_agent_position_observation(self)
                        + self.env.get_observation(self),
                    ),
                )
        else:
            parsed_actions = self.output_parser.parse(response.lower())
            action_dicts = []
            for action_name, action_arg in parsed_actions:
                if action_name in ["move_ahead", "turn_left", "turn_right", "stop"]:
                    action_dicts.append(
                        dict(action_name=self._justify_action_name(action_name))
                    )
                elif action_name in ["pick_up", "drop", "open", "close"]:
                    obj_name = action_arg.strip("'").strip('"')
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
                            dict(action_name=self._justify_action_name(action_name))
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

    def fetch_prompt_response(self, prompt):
        prompts = deepcopy(self.chat_history)
        if len(self.error_cache) > 0:
            prompts += deepcopy(self.error_cache)
        # prompts.append({"role": "system", "content": str(prompt)})

        self.history_all[f"{len(self.history_all)}"] = prompts[1:]

        # self.chat_history.append({"role": "system", "content": str(prompt)})
        response = self.llm(messages=create_llm_prompts(messages=prompts))
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
class RearrangeAgentReact(RearrangeAgent):
    def parse_response(self, response):
        if not self.status["started"]:
            if any([not_tok in response for not_tok in ["not", "don't"]]):
                return dict(
                    response=response,
                    feedback=self.env.feedback_builder.build(
                        "intro",
                        max_view_steps=self.max_view_distance / self.step_size,
                        degree=floor(self.aov / 2),
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
                        original_state=self.env.target_pose_description,
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
