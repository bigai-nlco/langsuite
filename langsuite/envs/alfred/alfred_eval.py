# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import os
from pathlib import Path

from langsuite.constants import WORKSPACE_PATH
from langsuite.utils.logging import logger
import json
import os
import os.path
import re
import sys
from pathlib import Path

ProjectPath = Path(__file__).parent.parent.parent.parent
print(ProjectPath)


def eval(folder_name):
    idx = 0
    true_number = 0
    conditioned_success = 0.0
    result = {}
    # p = "./scripts/test/alfred_test/gpt3.5-loc/"
    # print(len(os.listdir(p)))
    for fn in os.listdir(folder_name):
        idx += 1
        if not os.path.isfile(folder_name + fn):
            print(fn)
            continue
        with open(folder_name + fn, "r") as file:
            lines = file.readlines()

            data = []
            for line in lines:
                try: 
                    if line != "\n":
                        data.append(json.loads(line))
                except:
                    print("")
                    # print(fn, line)
            # data = [json.loads(line) for line in lines]
            if len(data) > 3:
                last_second = data[-2]
                last_third = data[-3]
                # print(last_second, last_third)
                if "content" in last_second and "[SUCCESS]" in last_second["content"]:
                    true_number += 1
                    conditioned_score = float(last_third["content"])
                    # print(conditioned_score)
                    if conditioned_score > 1.0:
                        # print(fn)
                        conditioned_success += 1.0
                    else:
                        conditioned_success += conditioned_score
                if "content" in last_third and last_third["content"].startswith("0."):
                    conditioned_score = float(last_third["content"])
                    # print(conditioned_score)
                    if conditioned_score > 1.0:
                        # print(fn)
                        conditioned_success += 1.0
                    else:
                        conditioned_success += conditioned_score
                if "content" in last_third and last_third["content"].startswith("2."):
                    true_number += 1
                    conditioned_score = float(last_third["content"])
                    # print(conditioned_score)
                    if conditioned_score > 1.0:
                        # print(fn)
                        conditioned_success += 1.0
                    else:
                        conditioned_success += conditioned_score
        
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
    dev_path = os.path.join(ProjectPath, "data", "alfred", "dev.json")
    with open(dev_path, "w", encoding="utf-8") as f:
        print(dev_path)
        failed_dict = {"dev": failed}
        json.dump(failed_dict, f)


def main():
    # stage = "iqa_data_folder/"
    # a = os.path.join(ProjectPath, stage, newnewest_folder, "console-logs/")
    path = (
        "./scripts/test/alfred_test/low_level_loc_reflexion/"
    )
    if len(sys.argv) == 2:
        path = sys.argv[1]
    if not os.path.exists(path):
        print("Dir not exist!")
        sys.exit()
    # locate_fail(path)
    eval(path)


if __name__ == "__main__":
    main()
