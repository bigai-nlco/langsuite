# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import logging
import os
from pathlib import Path

from langsuite.cli.cmd_cli import CMDClient

__all__ = ["logger"]

logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("PIL.PngImagePlugin").disabled = True


class Logger:
    def __init__(
        self,
        log_level: int = logging.DEBUG,
        log_file: str = "",
        use_cmd: bool = False,
        console_logging=True,
    ) -> None:
        self._logger = logging.getLogger("LangSuitE")
        self._logger.setLevel(log_level)
        self.log_level = log_level
        self.logFormatter = logging.Formatter(
            "[%(name)s %(levelname)s][%(asctime)s] %(message)s"
        )
        self._cmd = None

        if use_cmd:
            self._cmd = CMDClient(log_level=log_level)

        if console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.logFormatter)
            console_handler.setLevel(log_level)
            self._logger.addHandler(console_handler)
        self.console_logging = console_logging

    @property
    def has_cmdline_interface(self):
        return self._cmd is not None

    def setLevel(self, level):
        self.log_level = logging.getLevelName(
            level.upper() if type(level) == str else level
        )
        self._logger.setLevel(self.log_level)
        for hdlr in self._logger.handlers:
            hdlr.setLevel(self.log_level)

    def set_cmd_client(self, cmd_cli: CMDClient, disable_console_logging=True):
        self._cmd = cmd_cli
        if disable_console_logging:
            for hdlr in self._logger.handlers:
                if isinstance(hdlr, logging.StreamHandler):
                    self._logger.removeHandler(hdlr)

            self.console_logging = not disable_console_logging

    def set_log_file(self, log_file):
        if log_file and len(log_file) > 0:
            log_dir = Path(log_file).parent
            os.makedirs(log_dir, exist_ok=True)
            fileHandler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fileHandler.setFormatter(self.logFormatter)
            fileHandler.setLevel(self.log_level)
            self._logger.addHandler(fileHandler)

    def close(self):
        if self._cmd_log_file:
            self._cmd_log_file.close()

        if self._cmd:
            self._cmd.close()

    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)
        # if self._cmd and not self.console_logging:
        #     self._cmd.debug(msg)

    def error(self, msg):
        self._logger.error(msg)
        if self._cmd and not self.console_logging:
            self._cmd.error(msg)

    def warn(self, msg):
        self._logger.warning(msg)
        if self._cmd and not self.console_logging:
            self._cmd.warn(msg)

    def user_input(self):
        if self.has_cmdline_interface:
            return self._logger.user_input()

    def emit(self, message):
        if self.has_cmdline_interface:
            self._cmd.step(message)

        self._logger.info(message)

    def robot_emit(self, message_or_streamer, name="Robot", action="chat"):
        if self.has_cmdline_interface:
            if type(message_or_streamer) == str:
                self._cmd.step(
                    dict(
                        content=message_or_streamer,
                        role="assistant",
                        name=name,
                        action=action,
                    )
                )
            else:
                self._cmd.step(
                    dict(role="assistant", name=name, action=action),
                    stream=message_or_streamer,
                )
        self._logger.info(f"{name}: [{action.upper()}] {message_or_streamer}")


logger = Logger()
