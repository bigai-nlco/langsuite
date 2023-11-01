# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import os
from pathlib import Path

from langsuite.constants import WORKSPACE_PATH
from langsuite.utils.logging import logger


def eval(folder_name):
    idx = 0
    true_number = 0
    conditioned_success = 0.0
    result = {}
    for fn in os.listdir(folder_name):
        with open(folder_name + fn, "r") as file:
            data = [json.loads(line) for line in file]
            if len(data) > 3:
                last_second = data[-2]
                last_third = data[-3]
                if "content" in last_second and "[SUCCESS]" in last_second["content"]:
                    true_number += 1
                if "content" in last_third and last_third["content"].startswith("0."):
                    conditioned_success += float(last_third["content"])
        idx += 1
    logger.debug(
        "right number: {}, total number: {}, accuracy: {}, conditioned_success: {}".format(
            true_number,
            idx,
            round(float(true_number / idx), 4),
            round(float(conditioned_success / idx), 4),
        )
    )


def locate_fail(folder_name):
    failed = []
    for fn in os.listdir(folder_name):
        with open(folder_name + fn, "r") as file:
            data = [json.loads(line) for line in file]
            if len(data) > 0 and "content" in data[0]:
                data_path = data[0]["content"]
                for i in data:
                    if "content" in i and "Feedback: Action failed." in i["content"]:
                        failed.append(data_path)
                        # print(data_path)
                        break
    dev_path = Path(WORKSPACE_PATH, "data", "alfred", "dev.json")
    with open(dev_path, "w", encoding="utf-8") as f:
        logger.debug(dev_path)
        failed_dict = {"dev": failed}
        json.dump(failed_dict, f)
