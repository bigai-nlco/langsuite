# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
import argparse
import pathlib
import random
import sys
import traceback

from click import Path
import langsuite
from langsuite.envs.alfred.alfred_task import AlfredTask, load_data
import langsuite.server
import langsuite.webui
from datetime import datetime
from langsuite.utils.logging import logger
from langsuite.utils import io_utils
from langsuite.cli.cmd_cli import GameEndException, CMDClient


def create_from_config(config_path, task_data):
    config = io_utils.read_config(config_path)
    logger.info(config)
    task = task = AlfredTask.create(config, task_data=task_data)

    return task


def run_cmd_cli(task_or_config=None, verbose=False, task_data=None, stage=None, model="gpt3.5"):
    cmd_cli = CMDClient()
    task_path = "--".join(task_data['task_path'].split("/")[-2:])
    cmd_cli.set_cmd_log_file(
        log_file=f"alfred_{stage}/{model}/{task_path}"+".jl"
    )
    logger.set_cmd_client(cmd_cli, disable_console_logging=not verbose)
    logger.emit(task_data["task_path"])
    cmd_cli.start()
    # print(task_data["task_path"])
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
    # args = parser.parse_args()=
    stage = "test"
    ## 设置model 改变存储位置
    model = "temperature0"
    ## 设置数据路径
    data_path = "/home/wangmengmeng/workplace/data/alfred/alfred_test"

    ## conig path
    config_path = "/home/wangmengmeng/workplace/gitlab/sim2text/configs/alfred_cfg.yml"
    logger.set_log_file(
        f"alfred_{stage}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')}"
    )
    logger.setLevel("ERROR")
    index = 0
    # memory_path = "/home/wangmengmeng/workplace/gitlab/sim2text/scripts/test/alfred_test/memory_1.txt"

    tasks_data = load_data(data_path, stage)
    # print(len(tasks_data))
    random.seed(0)
    random.shuffle(tasks_data)
    count = 0

    for task_data in tasks_data[0:10]:
        print("***********", count+index)
        count+=1
        config_path = (
            config_path
        )
        # run_cmd_cli(
        #         task_or_config=config_path,
        #         verbose=False,
        #         task_data=task_data,
        #         stage=stage,
        #     )
        try:
            run_cmd_cli(
                task_or_config=config_path,
                verbose=False,
                task_data=task_data,
                stage=stage,
                model=model,
            )
        except Exception as e:
            traceback.print_exc()
            print(print("***********",count+index))
            continue


if __name__ == "__main__":
    main()
