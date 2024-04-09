# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations
from email import message

import json
import logging
import os
from pathlib import Path
from sys import stdout
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from langsuite.suit import Message

WELCOME_MSG = """
        __                      _____       _ __        ______
       / /   ____ _____  ____ _/ ___/__  __(_) /_  _   / ____/
      / /   / __ `/ __ \/ __ `/\__ \/ / / / / __/ (_) / __/
     / /___/ /_/ / / / / /_/ /___/ / /_/ / / /_      / /___
    /_____/\__,_/_/ /_/\__, //____/\__,_/_/\__/     /_____/
                      /____/

"""

HELP_MSG = [
    ("Load <task-id>", "Load task with task-id"),
    ("Help", "Print help message."),
    ("Quit", "Exit LangSuitE"),
]

COLORED_ROLE_STYLES = dict(
    system={"bold": True, "color": "yellow"},
    user={"bold": True, "color": "blue"},
)


class GameEndException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CMDClient:
    """
    Colors: https://rich.readthedocs.io/en/stable/appendix/colors.html
    """

    console_cfg: dict[str, Any] = dict(soft_wrap=True, markup=False, emoji=False, highlight=True)

    def __init__(self, *, log_level: int = logging.INFO, log_file=None, verbose=False) -> None:
        super().__init__()
        self._cmd_log_file = None
        if log_file and len(log_file) > 0:
            self.set_file(log_file)
        self._console = Console(**CMDClient.console_cfg)
        self._handler = RichHandler(console=self._console)
        self._handler.setLevel(log_level)
        
        self.cache = []
        self.verbose = verbose

    def set_file(self, log_file):
        if self._cmd_log_file:
            self._cmd_log_file.close()
        log_dir = Path(log_file).parent
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(log_file, "w", encoding="utf-8")
        self._cmd_log_file = log_file
        
    def print(self, message='', **kwargs):
        if self.verbose:
            self._console.print(message, **kwargs)

    def clear(self):
        if not self.verbose:
            self._console.clear()

    def end_task(self):
        self.print()
        self.print("Bye!", style="bold yellow")
        if self._cmd_log_file:
            self._cmd_log_file.close()

    def agent_step(self, message: Message, user_input = False, stream=False):
        try:
            if message.role in {"system", "function"}:
                if message.to:
                    self.print(
                        f"System (→ name): ",
                        style="bold cyan",
                        end="",
                    )
                else:
                    self.print("System: ", style="bold cyan", end="")
                self.print(message.stripped_content)
            elif message.role == "assistant":
                self.render_chatbot(
                    message.stripped_content,
                    name=message.name,
                    action=message.action,
                )
            if self._cmd_log_file:
                self._cmd_log_file.write(
                    json.dumps(message.dump_dict, sort_keys=True) + "\n"
                )
            if user_input:
                inp = self.user_input()
                if self._cmd_log_file:
                    self._cmd_log_file.write(
                        json.dumps(dict(role="user", content=inp), sort_keys=True)
                        + "\n"
                    )

                return inp
        except (KeyboardInterrupt, EOFError) as ex:
            raise GameEndException()        

    def env_step(self):
        pass

    def print_help(self):
        self.print("Help Info:")
        self.print('"Ctrl + C" or "Ctrl + D" to exit.')
        self._console.rule("Commands", style="bold yellow")
        for k, h in HELP_MSG:
            self.print(" " * 4 + "{:15}\t{:60}".format(k, h))
        self._console.rule(".", style="bold yellow")

    def user_input(self):
        try:
            self.print("User: ", style="bold green", end="")
            user_msg = self._console.input()
            return user_msg
        except UnicodeDecodeError as ex:
            self.print(
                f"Invalid input. Got UnicodeDecodeError: {ex}\nPlease try again."
            )
        return self.user_input()

    def reset(self):
        if self.verbose:
            self.clear()
        self.print(WELCOME_MSG)

    def render_chatbot(
        self, generator, name="Bot", action="chat", to="", stream: bool = True
    ):
        action = action.lower()
        if action == "chat" and len(to) > 0:
            self.print(f"Assistant ({name} → {to})", style="bold blue", end="")
        else:
            self.print(f"Assistant ({name})", style="bold blue", end="")
        if action == "thought":
            self.print(" THOUGHT", style="bold yellow", end="")
        elif action == "act":
            self.print(" ACT", style="bold cyan", end="")
        elif action != "chat":
            raise ValueError(f"Unknown action type: {action}")
        self.print(": ", style="bold blue", end="")
        if type(generator) == str:
            self.print(generator)

if __name__ == "__main__":
    cmd = CMDClient()
    cmd.reset()
