# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import logging
import os
from pathlib import Path

from langsuite.cli.cmd_cli import CMDClient

from rich.logging import RichHandler

__all__ = ["logger"]

logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("PIL.PngImagePlugin").disabled = True

class LoggerSingleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(LoggerSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class Logger(logging.getLoggerClass(), metaclass=LoggerSingleton):
    def __init__(
        self,
        log_level: int = logging.INFO,
        use_cmd: bool = False,
        console_logging=True,
    ) -> None:
        super().__init__(name='LangSuitE')
        self.setLevel(log_level)
        self.logFormatter = logging.Formatter(
            "[%(name)s %(levelname)s][%(asctime)s %(filename)s:%(lineno)d - %(funcName)s] %(message)s"
        )
        self._cmd = None

        if use_cmd:
            self._cmd = CMDClient(log_level=log_level)

        if console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.logFormatter)
            console_handler.setLevel(log_level)
            self.addHandler(console_handler)
        self.console_logging = console_logging

    @property
    def has_cmdline_interface(self):
        return self._cmd is not None

    def setLevel(self, level):
        self.log_level = logging.getLevelName(
            level.upper() if type(level) == str else level
        )
        for hdlr in self.handlers:
            hdlr.setLevel(self.log_level)

    def set_cmd_client(self, cmd_cli: CMDClient, disable_console_logging=True):
        self._cmd = cmd_cli
        if disable_console_logging:
            for hdlr in self.handlers:
                if isinstance(hdlr, type(cmd_cli._handler)):
                    self.removeHandler(hdlr)

            self.console_logging = not disable_console_logging
        self.addHandler(cmd_cli._handler)

    def set_log_file(self, log_file):
        if log_file and len(log_file) > 0:
            log_dir = Path(log_file).parent
            os.makedirs(log_dir, exist_ok=True)
            fileHandler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fileHandler.setFormatter(self.logFormatter)
            fileHandler.setLevel(self.log_level)
            self.addHandler(fileHandler)

logger = Logger()
