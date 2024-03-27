# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations
import json
from typing import Sequence
import requests

import yaml

from langsuite.suit.message import Message


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f.read())
        return config