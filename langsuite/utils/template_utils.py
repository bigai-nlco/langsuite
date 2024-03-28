# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import os
import random
import re

from langsuite.utils.logging import logger


def split(data: dict):
    def split_t(template: list):
        return [
            {"template": content, "params": re.findall(r"\{([^}]+)\}", content)}
            for content in template
        ]

    return {
        act_name: {
            status: split_t(inner_value) for status, inner_value in inner_dict.items()
        }
        for act_name, inner_dict in data.items()
    }
