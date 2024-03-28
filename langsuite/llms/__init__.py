from __future__ import annotations

import importlib
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

import yaml

from langsuite.constants import WORKSPACE_PATH
from langsuite.llms.langchain import (
    AIMessage,
    BaseChatModel,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    BaseMessage,
)
from langsuite.suit.message import Message
from langsuite.utils.logging import logger


def _init_llm_config(llm_type):
    """Initialize LLM config"""
    if "OpenAI" in llm_type:
        if os.getenv("OPENAI_API_KEY"):
            return {
                "llm_type": llm_type,
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
            }
    with open(Path(WORKSPACE_PATH, "api.config.yml"), "r") as f:
        api_config = yaml.safe_load(f)
        return api_config


def create_llm(config=None) -> Optional[BaseChatModel]:
    """Create a LLM from config"""
    logger.info("create llm with config: %s", config)
    if config is None:
        config = dict()
    llm_cfg = _init_llm_config(config.get("llm_type", ""))
    if not llm_cfg:
        logger.error("LLM config not found.")
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

def create_llm_prompts(messages: Union[Message, BaseMessage, list]) -> list[BaseMessage]:
    """Convert a list of messages to a langchain prompt"""
    if type(messages) in {BaseMessage, Message}:
        messages = [messages]
    result = []
    for msg in messages: # type: ignore
        if isinstance(msg, BaseMessage):
            result.append(msg)
        if isinstance(msg, Message):
            if msg.role in {"assistant", "ai"}:
                result.append(AIMessage(content=msg.raw_content))
            elif msg.role in {"human", "user"}:
                result.append(HumanMessage(content=msg.raw_content))
            elif msg.role in {"system"}:
                result.append(SystemMessage(content=msg.raw_content))
            elif msg.role in {"function"}:
                result.append(FunctionMessage(content=msg.raw_content, name="environment"))
    return result

def process_llm_responses(responses, agent_name) -> List[dict]:
    if isinstance(responses, BaseMessage):
        responses = [responses]
    result = []
    for res in responses:
        if isinstance(res, BaseMessage):
            role = 'assistant'
        if isinstance(res, HumanMessage):
            role = 'human'
        elif isinstance(res, SystemMessage) or isinstance(res, FunctionMessage):
            role = 'system'
        result.append({'role': role, 'content': res.content})
    
    if len(result):
        return result
    raise ValueError(f"Unable to process llm results: {responses}")
