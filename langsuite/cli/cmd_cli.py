# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from rich.console import Console

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

    console_cfg = dict(soft_wrap=True, markup=False, emoji=False, highlight=True)

    def __init__(self, *, log_level: int = logging.DEBUG, log_file=None) -> None:
        self.log_level = log_level
        if log_file and len(log_file) > 0:
            log_dir = Path(log_file).parent
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            log_file = open(log_file, "w")
            self.console = Console(**CMDClient.console_cfg)
        else:
            self.console = Console(**CMDClient.console_cfg)

        self._cmd_log_file = log_file

    def set_cmd_log_file(self, log_file):
        if self._cmd_log_file:
            self._cmd_log_file.close()
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)
        self._cmd_log_file = open(log_file, "w+", encoding="utf-8")

    def reset(self):
        self.clear()

    def clear(self):
        self.console.clear()

    def close(self):
        self.console.print()
        self.console.print("Bye!", style="bold yellow")
        if self._cmd_log_file:
            self._cmd_log_file.close()

    def info(self, message: str):
        if self.log_level <= logging.INFO:
            self.console.log("[INFO] ", style="bold", end="")
            self.console.print(message)

    def error(self, message: str):
        if self.log_level <= logging.ERROR:
            self.console.log("[ERROR] ", style="bold red", end="")
            self.console.print(message)

    def debug(self, message: str):
        if self.log_level <= logging.DEBUG:
            self.console.log("[DEBUG] ", style="bold bright_black", end="")
            self.console.print(message)

    def warn(self, message: str):
        if self.log_level <= logging.WARNING:
            self.console.log("[WARNING] ", style="bold yellow", end="")
            self.console.print(message)

    def step(self, message=None, user_input: bool = False, stream=False):
        """
        Args:
            message: dict(
                role: ["system"|"assistant"],
                content: str,
                name: str,
                action: str
            )

            stream: bool or Generator
        """

        try:
            if message:
                if type(message) == list:
                    for msg in message:
                        self.step(msg, user_input=False, stream=stream)
                else:
                    if type(message) == str:
                        message = {"role": "system", "content": message}

                    if message["role"] == "system":
                        if len(message.get("to", "")) > 0:
                            self.console.print(
                                f"System (→ {message['to']}): ",
                                style="bold cyan",
                                end="",
                            )
                        else:
                            self.console.print("System: ", style="bold cyan", end="")
                        self.console.print(message["content"])
                    elif message["role"] == "assistant":
                        if stream:
                            self.render_chatbot(
                                stream,
                                name=message.get("name", "Robot"),
                                action=message.get("action", "chat"),
                            )
                        else:
                            self.render_chatbot(
                                message["content"],
                                name=message.get("name", "Robot"),
                                action=message.get("action", "chat"),
                            )
                    if self._cmd_log_file:
                        self._cmd_log_file.write(
                            json.dumps(message, sort_keys=True) + "\n"
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

    def print_help(self):
        self.console.print("Help Info:")
        self.console.print('"Ctrl + C" or "Ctrl + D" to exit.')
        self.console.rule("Commands", style="bold yellow")
        for k, h in HELP_MSG:
            self.console.print(" " * 4 + "{:15}\t{:60}".format(k, h))
        self.console.rule(".", style="bold yellow")

    def cmd_input(self):
        try:
            cmd_msg = self.console.input(prompt="> ")
        except UnicodeDecodeError as ex:
            self.console.print_exception(show_locals=True)
            self.error(
                f"Invalid input. Got UnicodeDecodeError: {ex}\nPlease try again."
            )
        except KeyboardInterrupt:
            raise GameEndException()

        cmd = cmd_msg.strip().split(" ")
        if cmd[0].upper() == "LOAD":
            pass
        elif cmd[0].upper() == "HELP":
            self.print_help()
        else:
            raise NotImplementedError

        cmd[0] = cmd[0].upper()
        return cmd

    def user_input(self):
        try:
            self.console.print("User: ", style="bold green", end="")
            user_msg = self.console.input()
        except UnicodeDecodeError as ex:
            self.error(
                f"Invalid input. Got UnicodeDecodeError: {ex}\nPlease try again."
            )
        return user_msg

    def start(self):
        self.console.print(WELCOME_MSG)

    def render_chatbot(
        self, generator, name="Bot", action="chat", to="", stream: bool = True
    ):
        action = action.lower()
        if action == "chat" and len(to) > 0:
            self.console.print(f"Assistant ({name} → {to})", style="bold blue", end="")
        else:
            self.console.print(f"Assistant ({name})", style="bold blue", end="")
        if action == "thought":
            self.console.print(" THOUGHT", style="bold yellow", end="")
        elif action == "act":
            self.console.print(" ACT", style="bold cyan", end="")
        elif action != "chat":
            raise ValueError(f"Unknown action type: {action}")
        self.console.print(": ", style="bold blue", end="")
        if type(generator) == str:
            self.console.print(generator)


if __name__ == "__main__":
    cmd = CMDClient()
    cmd.start()
