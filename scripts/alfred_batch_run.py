# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import logging
import os
import random
from tabnanny import verbose
import traceback
from datetime import datetime
from pathlib import Path
from langsuite.cli import cmd_cli

from langsuite.cli.cmd_cli import CMDClient, GameEndException
from langsuite.constants import WORKSPACE_PATH
from langsuite.tasks.alfred_v0.task import AlfredTask_V0
from langsuite.utils import io_utils
from langsuite.utils.logging import logger

random.seed(0)

def create_from_config(config_path, task_data, cli):
    config = io_utils.read_config(config_path)
    task = AlfredTask_V0.create(config, task_data=task_data, cmd_cli=cli)

    return task


def run_cmd_cli(
    task_or_config, task_data, stage, cmd_cli: CMDClient, set_logging=True,
) -> int:
    task_id = task_data['path'].split("/")[-2]
    task_data['log_file'] = f"alfred_{stage}/{task_id}"+".jl"
    
    task = create_from_config(task_or_config, task_data, cmd_cli)

    answer = 0
    try:
        log_file=f"logs/console_logs/alfred_{stage}/{task_id}.jl"
        cmd_cli.set_file(log_file)
        if set_logging:
            logger.set_cmd_client(cmd_cli, disable_console_logging=True)
        cmd_cli.reset()
        answer_dict = task.run()
        #HACK
        answer = next(iter(answer_dict.values()))
    except GameEndException:
        pass
    finally:
        cmd_cli.end_task()
    
    return answer

def main():
    stage = "train_debug"
    data_path = "./data/alfred/"
    config_path = "./configs/alfred_debug_cfg.yml"
    logger.set_log_file(
        f"logs/alfred_{stage}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')}"
    )
    logger.setLevel("ERROR")
    index = 0
    # memory_path = "./scripts/test/alfred_test/memory_1.txt"
    cmd_cli = CMDClient(verbose=False, log_level=logging.ERROR)

    tasks_data = AlfredTask_V0.load_data(data_path, stage)
    # print(len(tasks_data))
    random.shuffle(tasks_data)
    count = 0
    right = 0
    for task_data in tasks_data:
        print("***********", count+index)
        count+=1
        config_path = (
            config_path
        )
        try:
            right += run_cmd_cli(
                task_or_config=config_path,
                task_data=task_data,
                stage=stage,
                cmd_cli=cmd_cli,
            )
        except Exception as e:
            traceback.print_exc()
            print(print("***********",count+index))
            continue
    result_str = "right: {}/{} = {}".format(right, count, 1.0 * right / count)
    cmd_cli._console.print(result_str)

if __name__ == "__main__":
    main()
