from __future__ import annotations

import importlib
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml

from langsuite.constants import WORKSPACE_PATH
from langsuite.llms.langchain import (
    AIMessage,
    BaseChatModel,
    HumanMessage,
    SystemMessage,
)
from langsuite.utils.logging import logger


def _init_llm_config(llm_type="ChatOpenAI"):
    """Initialize LLM config"""
    if "OpenAI" in llm_type:
        if os.getenv("OPENAI_API_KEY"):
            return {
                "llm_type": llm_type,
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
            }
        with open(Path(WORKSPACE_PATH, "api.config.yml"), "r") as f:
            api_config = yaml.safe_load(f)
        try:
            api_cfg = [cfg for cfg in api_config if cfg["llm_type"] == llm_type][0]
        except IndexError:
            api_cfg = None

        return api_cfg


def create_llm(config=None) -> Optional[BaseChatModel]:
    """Create a LLM from config"""
    if config is None:
        config = dict()
    llm_cfg = _init_llm_config(config.get("llm_type", "ChatOpenAI"))
    if not llm_cfg:
        return None

    class_cfg = deepcopy(llm_cfg)
    class_cfg.update(config)
    logger.info(class_cfg)
    llm_type = class_cfg.pop("llm_type", None)
    if not llm_type:
        logger.info(f"Using default LLM: \"{class_cfg['llm_type']}\"")
        llm_type = class_cfg["llm_type"]

    llm = getattr(importlib.import_module("langsuite.llms.langchain"), llm_type)
    if llm:
        return llm(**class_cfg)


def create_llm_prompts(messages):
    """Convert a list of messages to a langchain prompt"""
    if type(messages) == list:
        return [create_llm_prompts(message) for message in messages]

    if (
        isinstance(messages, AIMessage)
        or isinstance(messages, HumanMessage)
        or isinstance(messages, SystemMessage)
    ):
        return messages

    if messages["role"] in ["assistant", "ai"]:
        return AIMessage(content=messages["content"])

    if messages["role"] in ["human", "user"]:
        return HumanMessage(content=messages["content"])

    if messages["role"] in ["system"]:
        return SystemMessage(content=messages["content"])


def process_llm_results(llm_results):
    if type(llm_results) == list:
        return [process_llm_results(llm_result) for llm_result in llm_results]

    if type(llm_results) == str:
        return llm_results

    if isinstance(llm_results, AIMessage):
        return llm_results.content

    raise ValueError(f"Unable to process llm results: {llm_results}")
