# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import re
from abc import ABC

from pydantic import BaseModel

from langsuite.utils.registry import Registry

ACTION_REGISTERY = Registry("actions")


class BaseAction(ABC):
    def __init__(self, env=None, agent=None, **kwargs) -> None:
        w_s = re.findall(r"[A-Z][^A-Z]*", self.__class__.__name__.replace("Action", ""))
        self.name = "_".join([w.upper() for w in w_s])

        self.env = env
        self.agent = agent
        self.kwlist = []

        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        if env:
            self.feedback_builder = env.get_feedback_builder()

    def step(self, **kwargs):
        if not self.check_validity():
            return ActionFeedback(
                agent=self.agent.id,
                success=False,
                feedback=self.feedback_builder.build("InvalidAction", action=self.name),
            )
        args = []
        if "action_args" in kwargs:
            args = kwargs.pop("action_args")
        idx = 0
        for kw in self.kwlist:
            if kw not in kwargs and len(args) > idx:
                kwargs[kw] = args[idx]
                idx += 1
        return self.exec(**kwargs)

    def check_validity(self, **kwargs) -> bool:
        return True

    def exec(self, **kwargs):
        raise NotImplementedError


class ActionFeedback(BaseModel):
    success: bool
    feedback: str


class BabyAIActionFeedback(BaseModel):
    success: bool
    feedback: str
    task_success: bool
