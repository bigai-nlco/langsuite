# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import argparse
from datetime import datetime
from tabnanny import verbose

import langsuite
import langsuite.server
from langsuite.suit import make_task
import langsuite.webui
from langsuite.cli.cmd_cli import CMDClient, GameEndException
from langsuite.utils import io_utils
from langsuite.utils.logging import logger


def run_cmd_cli(config_path, verbose=False):
    cmd_cli = CMDClient(
        log_file=f"logs/console_logs/{datetime.now().strftime('console-%Y-%m-%d_%H-%M-%S.jl')}",
        verbose=verbose
    )
    logger.set_cmd_client(cmd_cli, disable_console_logging=not verbose)
    cmd_cli.reset()

    config = io_utils.read_config(config_path)
    task = make_task(config, cmd_cli=cmd_cli)
    try:
        task.run()
    except GameEndException:
#        cmd_cli.console.log()
        pass
    finally:
        cmd_cli.end_task()


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
        config = io_utils.read_config(args.config)
        task = make_task(config)
        return langsuite.server.serve(task, args)
    elif cmd_action in ["cmd", "task", "config", "load"]:
        run_cmd_cli(args.config, verbose=args.verbose)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
