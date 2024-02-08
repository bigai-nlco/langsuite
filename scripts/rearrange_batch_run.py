# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations
from pathlib import Path
import random

import traceback
from datetime import datetime

from langsuite.cli.cmd_cli import CMDClient, GameEndException
from langsuite.envs.rearrange.rearrangement_task import RearrangementTask, load_data
from langsuite.utils import io_utils
from langsuite.utils.logging import logger


def run_cmd_cli(task_or_config=None, verbose=False, task_data=None, stage=None, scene=None, index=None):
    cmd_cli = CMDClient()
    task_data['log_file'] = f"{scene}-{index}.jl"
    try:
        config = io_utils.read_config(task_or_config)
        logger.info(config)
        task = RearrangementTask.create(config, task_data=task_data)
        cmd_cli.set_cmd_log_file(
            log_file=f"rearrange/{Path(config['template']).stem}/console-logs/{scene}-{index}.jl"
        )
        logger.set_cmd_client(cmd_cli, disable_console_logging=not verbose)
        cmd_cli.start()
        task.run()
    except GameEndException:
        pass
    cmd_cli.close()


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("action", default="cmd", help="Langsuite actions")
    # parser.add_argument("config", default=None, help="Config file path")
    # parser.add_argument("--verbose", default=False, action="store_true")
    # parser.add_argument("--log-level", default="debug", type=str)
    # args = parser.parse_args()
    stage = "val"
    logger.set_log_file(f"rearrange_{stage}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')}")
    logger.setLevel("ERROR")

    tasks_data = load_data("", stage)
    random.seed(0)
    random.shuffle(tasks_data)
    for task_data in tasks_data[:50]:
        scene = task_data["scene"]
        index = task_data["index"]
        config_path = "./configs/rearrange_cfg.yml"

        try:
            run_cmd_cli(
                task_or_config=config_path,
                verbose=False,
                task_data=task_data,
                stage=stage,
                scene=scene,
                index=index,
            )
        except Exception as e:
            traceback.print_exc()
            print("scene:", scene, "index:", index)
            continue

if __name__ == "__main__":
    main()
