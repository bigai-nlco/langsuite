# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from langsuite.cli.cmd_cli import CMDClient, GameEndException
from langsuite.constants import WORKSPACE_PATH
from langsuite.envs.iqa.iqa_task import IqaTask
from langsuite.utils import io_utils
from langsuite.utils.logging import logger


def create_from_config(config_path, task_data):
    config = io_utils.read_config(config_path)
    logger.info(config)
    task = IqaTask.create(config, task_data=task_data)

    return task


def run_cmd_cli(
    task_or_config=None, verbose=False, task_data=None, stage=None, index=None, qt=None
):
    cmd_cli = CMDClient()
    cmd_cli.set_cmd_log_file(
        log_file=f"{stage}/console-logs/data_type_{qt}_idx_{index}_.jl"
    )
    logger.set_cmd_client(cmd_cli, disable_console_logging=not verbose)
    cmd_cli.start()
    task_data['log_file'] = f"data_type_{qt}_idx_{index}_.jl"

    if task_or_config:
        try:
            task = create_from_config(task_or_config, task_data)
            task.run()
        except GameEndException:
            pass
    else:
        while True:
            try:
                cmd = cmd_cli.cmd_input()
                if cmd[0] == "LOAD":
                    task = create_from_config(cmd[1])
                    task.run()
            except GameEndException:
                break
    cmd_cli.close()


def load_data(data_dir):
    iqa_data = json.load(open(data_dir))
    task_data = []
    for _id, world_data in enumerate(iqa_data):
        task_data.append(
            dict(
                name=f"Iqa:Iqa2DEnv:{_id}",
                data=dict(world_data=world_data[0]),
                task_definition="",
                inputs=[],
                targets=[],
                qa=world_data[1],
            )
        )
    return task_data


def main():
    stage = "iqa_data_folder"
    question_type = 0
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = Path(
        WORKSPACE_PATH,
        stage,
        "task_type_" + str(question_type) + "_" + current_time,
    )
    os.mkdir(folder_path)
    # a = c
    logger.set_log_file(
        f"{stage}/{current_time}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')}"
    )
    logger.setLevel("ERROR")
    file_path = Path(WORKSPACE_PATH, "data", "iqa","iqa_test", "iqa_test_1k.json")
    idx_lis = [446, 1818, 186, 1622, 1214, 752, 1020, 236, 1504, 58, 1292, 1630, 1454, 380, 476, 1394, 1720, 288, 1396, 1308, 1550, 1658, 794, 1994, 1406, 106, 1686, 1278, 256, 1796, 1224, 1128, 654, 1424, 820, 428, 48, 396, 1084, 494, 1056, 1578, 120, 1246, 352, 438, 1402, 1440, 762, 1520]
    #idx_lis = [23, 999, 100]
    tasks_data_all = load_data(file_path)[:1000]
    tasks_data = []
    for i in idx_lis:
        idx = int(i / 2 - 1)
        tasks_data.append(tasks_data_all[idx])
    


    index = 0
    for t in tasks_data:
        index += 1
        config_path = Path(WORKSPACE_PATH, "configs", "iqa_cfg.yml")
        try:
            run_cmd_cli(
                task_or_config=config_path,
                verbose=False,
                task_data=t,
                stage=f"{stage}/task_type_{question_type}_{current_time}",
                index=index,
                qt=question_type,
            )
        except Exception as e:
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
