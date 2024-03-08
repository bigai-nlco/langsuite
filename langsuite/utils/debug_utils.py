from __future__ import annotations
import json
from typing import Sequence

import requests

from langsuite.suit.message import Message
from langsuite.utils.logging import logger


def manual_request(messages: Sequence[Message]) -> list[dict]:
    str_messages = [msg.dump_dict for msg in messages]
    api_key = ""
    api_url = ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": "gpt-3.5-turbo", "messages": str_messages, "temperature": 0}
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        data = response.json()
        return [data["choices"][0]["message"]]
    else:
        error = f"Error: Received status code {response.status_code}"
        raise Exception(error)
