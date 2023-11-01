from __future__ import annotations

import json
import os
import re
from pathlib import Path


def eval(folder_name):
    idx = 0
    true_number = 0
    for fn in os.listdir(folder_name):
        with open(folder_name + fn, "r") as file:
            data = [json.loads(line) for line in file]
            idx += 1
            if len(data) > 4:
                one = data[-1]
                two = data[-2]
                three = data[-3]
                if (
                    "Obs" not in three["content"]
                    and "Feedback" not in three["content"]
                    and "content" in one
                    and "DONE!" in one["content"]
                    and "content" in two
                    and two["content"] == "Your are right!"
                ):
                    true_number += 1
    print("folder name is: ", folder_name)
    print(
        "right number: {}, total number: {}, accuracy: {}".format(
            true_number, idx, round(float(true_number / idx), 2)
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
            print(f"No matching folder found.")
        else:
            matching_folders.sort(reverse=True)
            newest_folder = matching_folders[0]
            return newest_folder
    return ""


ProjectPath = Path(__file__).parent.parent.parent.parent


def main():
    stage = "iqa_data_folder/"
    question_type = 2
    newnewest_folder = get_newest_folder(
        os.path.join(ProjectPath, stage), question_type
    )
    a = os.path.join(ProjectPath, stage, newnewest_folder, "console-logs/")
    eval(a)


if __name__ == "__main__":
    main()
