# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
import argparse
import os
import pathlib
import time
import traceback
import random
from click import Path
import langsuite
from langsuite.envs.babyai.babyai_task import BabyAITask, load_data
import langsuite.server
from langsuite.utils import logging
import langsuite.webui
from datetime import datetime
from langsuite.utils.logging import logger
from langsuite.utils import io_utils
from langsuite.cli.cmd_cli import GameEndException, CMDClient


def create_from_config(config_path, task_data):
    config = io_utils.read_config(config_path)
    logger.info(config)
    task = task = BabyAITask.create(config, task_data=task_data)

    return task


def run_cmd_cli(task_or_config=None, verbose=False, task_data=None, stage=None, trail=0):
    level = task_data["id"]
    seed = task_data["seed"]
    log_file=f"{stage}/{level}-{seed}-{str(trail)}.jl"
    task_data['log_file'] = f'{level}-{seed}-{str(trail)}.jl'
    # #XXX
    # print(task_data['log_file'])
    # return
    cmd_cli = CMDClient()
    if (os.path.exists(log_file)):# and os.path.getsize(log_file) > 0):
        print(f"Log file {log_file} exists")# and is not empty, skipping...")
        logger.info(f"Log file {log_file} exists")# and is not empty, skipping...")
        return
    cmd_cli.set_cmd_log_file(log_file=log_file)
    logger.set_cmd_client(cmd_cli, disable_console_logging=not verbose)
    cmd_cli.start()

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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("action", default="cmd", help="Langsuite actions")
    # parser.add_argument("config", default=None, help="Config file path")
    # parser.add_argument("--verbose", default=False, action="store_true")
    # parser.add_argument("--log-level", default="debug", type=str)
    # args = parser.parse_args()
    stage = "50"
#    stage_file = "babyai_gpt3.5_react_emmem"
    stage_file = "babyai_gpt3.5_reflexion_emmem"
#    stage_file = "babyai_gpt3.5_reflexion_examples"
#    stage_file = "babyai_gpt3.5_reflexion"
    logger.set_log_file(
        f"{stage_file}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')}"
    )
    logger.setLevel("INFO")
    
    tasks_data = load_data(10, stage)
    # print(tasks_data)
    ids = [
        "TestPutNextToCloseToDoor2",
        "TestUnblockingLoop",
        "TestLotsOfBlockers",
        "TestPutNextCloseToDoor",
        "TestGoToBlocked",
        "TestPutNextToBlocked",
        "TestPutNextToIdentical",
        "TestPutNextToCloseToDoor1",
    ]
#     random.seed(1)
# #    random.seed(0)
#     random.shuffle(tasks_data)
    task_list = set()
    with open('./langsuite/envs/babyai/task_list.txt', 'r') as lists:
        for l in lists.readlines():
            task_list.add(l.strip())
    for task_data in tasks_data:
        level = task_data["id"]
        seed = task_data['seed']
        if not f'{level}-{seed}' in task_list:
            print('Skip', f'{level}-{seed} not in list')
            # logging.logger._logger.info( f'Skip {level}-{seed} not in list')
            continue
        task_list.discard(f'{level}-{seed}')
        # ids.append(id)

        # seed = task_data["seed"]
        config_path = (
            "./configs/babyai_cfg.yml"
        )

        # config_path = Path(root, "configs", "rearrange_cfg.yml")
        # run_cmd_cli(task_or_config=config_path, verbose=False, task_data=task_data)
        # print("scene:", scene, "index:", index)
        # run_cmd_cli(
        #     task_or_config=config_path,
        #     verbose=False,
        #     task_data=task_data,
        #     stage=stage_file,
        # )
        # if id not in ids:
        #     continue
        for i in range(1):
            time.sleep(2)
            try:
                run_cmd_cli(
                    task_or_config=config_path,
                    verbose=False,
                    task_data=task_data,
                    stage=stage_file,
                    trail=i,
                )
                
            except Exception as e:
                traceback.print_exc()
                continue
    print('The following tasks are missing:')
    for t in task_list:
        print(t, end='\n')

if __name__ == "__main__":
    main()
