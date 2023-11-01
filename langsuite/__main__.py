# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import argparse
from datetime import datetime

import langsuite
import langsuite.server
import langsuite.webui
from langsuite.cli.cmd_cli import CMDClient, GameEndException
from langsuite.utils import io_utils
from langsuite.utils.logging import logger


def create_from_config(config_path):
    config = io_utils.read_config(config_path)
    logger.info(config)
    task = langsuite.make(config)

    return task


def run_cmd_cli(task_or_config=None, verbose=False):
    cmd_cli = CMDClient()
    cmd_cli.set_cmd_log_file(
        log_file=f"logs/console-logs/{datetime.now().strftime('console-%Y-%m-%d_%H-%M-%S.jl')}"
    )
    logger.set_cmd_client(cmd_cli, disable_console_logging=not verbose)
    cmd_cli.start()

    if task_or_config:
        try:
            task = create_from_config(task_or_config)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="cmd", help="Langsuite actions")
    parser.add_argument("config", default=None, nargs="?", help="Config file path")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--log-level", default="debug", type=str)
    args = parser.parse_args()

    logger.set_log_file(f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')}")
    logger.info(args)
    logger.setLevel(args.log_level)

    cmd_action = args.action.lower()
    if cmd_action == "webui":
        return langsuite.webui.run()

    if cmd_action == "serve":
        task = create_from_config(args.config)
        return langsuite.server.serve(task.env, args)
    elif cmd_action in ["cmd", "task", "config", "load"]:
        if args.config:
            run_cmd_cli(args.config, verbose=args.verbose)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
