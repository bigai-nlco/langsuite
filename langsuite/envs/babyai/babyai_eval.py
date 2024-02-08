from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from langsuite.constants import WORKSPACE_PATH
from langsuite.utils.logging import logger


def eval(folder_name):
    idx = 0
    true_number = 0
    result = {}
    root_path = Path(WORKSPACE_PATH, "./scripts/test/babyai_gpt3.5/console-logs/")
    # filter = os.listdir(path):
    for fn in os.listdir(root_path):
        if not fn.endswith("0.jl"):
            continue
        with open(folder_name + fn, "r") as file:
            task_des = fn.replace(".jl", "")
            task_des = task_des.split("-")
            task_name = task_des[0]
            task_seed = task_des[1]
            if task_name + "-" + task_seed not in result:
                result[task_name + "-" + task_seed] = 0

            data = [json.loads(line) for line in file]
            if len(data) > 2:
                last_second = data[-2]
                if "content" in last_second and "[SUCCESS]" in last_second["content"]:
                    result[task_name + "-" + task_seed] = 1
    for r in result:
        if result[r] == 1:
            true_number += 1
        idx += 1
    logger.debug(
        "right number: {}, total number: {}, accuracy: {}".format(
            true_number, idx, round(float(true_number / idx), 4)
        )
    )


def get_newest_folder(folder_path, question_type):
    pattern = (
        r"task_type_" + str(question_type) + "_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    )
    subfolders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]
    if not subfolders:
        print(f"there is no folder in {folder_path}. ")
    else:
        matching_folders = [
            folder for folder in subfolders if re.match(pattern, folder)
        ]

        if not matching_folders:
            logger.warn("No matching folder found.")
        else:
            matching_folders.sort(reverse=True)
            newest_folder = matching_folders[0]
            return newest_folder
    return ""


ProjectPath = Path(__file__).parent.parent.parent.parent


def main():
    # stage = "iqa_data_folder/"
    # a = os.path.join(ProjectPath, stage, newnewest_folder, "console-logs/")
    path = "./scripts/test/babyai_gpt3.5/console-logs/"
    if len(sys.argv) == 2:
        path = sys.argv[1]
    if not os.path.exists(path):
        print("Dir not exist!")
        sys.exit()
    eval(path)


if __name__ == "__main__":
    main()
