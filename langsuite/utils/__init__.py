# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations


class Counter:
    def __init__(self, name="") -> None:
        self.name = name
        self.cnt = 0

    def step(self):
        self.cnt += 1

    def __repr__(self) -> str:
        return f"{self.name}|{self.cnt}" if len(self.name) > 0 else str(self.cnt)

    def __str__(self) -> str:
        return self.__repr__()
