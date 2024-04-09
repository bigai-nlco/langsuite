# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import logging
import random
import traceback
from datetime import datetime

from langsuite.cli.cmd_cli import CMDClient, GameEndException
from langsuite.tasks.alfred_v0.task import AlfredTask_V0
from langsuite.utils import io_utils
from langsuite.utils.logging import logger

random.seed(0)
STAGE = "test"
DATA_PATH = "./data/alfred/"
METHOD = 'debug'
CONFIG_PATH = f"./configs/alfred_{METHOD}_cfg.yml"

def create_from_config(config_path, task_data, cli):
    config = io_utils.read_config(config_path)
    task = AlfredTask_V0.create(config, task_data=task_data, cmd_cli=cli)

    return task


def run_cmd_cli(
    task_or_config,
    task_data,
    cmd_cli: CMDClient,
    set_logging=True,
) -> int:
    task_id = task_data["path"].split("/")[-2]
    task_data["log_file"] = f"alfred_{STAGE}/{METHOD}/{task_id}.jl"

    task = create_from_config(task_or_config, task_data, cmd_cli)

    answer = 0
    try:
        log_file = f"logs/console_logs/alfred_{STAGE}/{task_id}.jl"
        cmd_cli.set_file(log_file)
        if set_logging:
            logger.set_cmd_client(cmd_cli, disable_console_logging=True)
        cmd_cli.reset()
        answer_dict = task.run()
        # HACK
        answer = next(iter(answer_dict.values()))
    except GameEndException:
        pass
    finally:
        cmd_cli.end_task()

    return answer

def main():
    logger.set_log_file(
        f"logs/alfred_{STAGE}/{METHOD}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')}"
    )
    logger.setLevel("INFO")
    index = 0
    # memory_path = "./scripts/test/alfred_test/memory_1.txt"
    cmd_cli = CMDClient(verbose=False, log_level=logging.WARNING)

    tasks_data = AlfredTask_V0.load_data(DATA_PATH, STAGE)
    # print(len(tasks_data))
    random.shuffle(tasks_data)
    count = 0
    right = 0
    for task_data in tasks_data[:200]:
        print("***********", count + index)
        count += 1
        try:
            right += run_cmd_cli(
                task_or_config=CONFIG_PATH,
                task_data=task_data,
                cmd_cli=cmd_cli,
            )
        except Exception:
            traceback.print_exc()
            print("***********", count + index)
            continue
    result_str = f"right: {right}/{count} = {1.0 * right / count}"
    cmd_cli._console.print(result_str)


if __name__ == "__main__":
    main()
