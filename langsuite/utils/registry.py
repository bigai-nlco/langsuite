# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from typing import Optional, Text


class Registry:
    def __init__(self, name: Text) -> None:
        self._name = name
        self._obj_map = {}

    def _register(self, name: Text, obj) -> None:
        # case insensitive
        if name.upper() in self._obj_map:
            raise RuntimeError(
                f"An object named {name} was already registered in {self._name}."
            )

        self._obj_map[name.upper()] = obj

    def register(self, obj=None, name: Optional[Text] = None):
        if obj is None:

            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._register(name, func_or_class)
                return func_or_class

            return deco

        if name is None:
            name = obj.__name__

        self._register(name, obj)

    def get(self, name):
        # case insensitive
        ret = self._obj_map.get(name.upper())
        if ret is None:
            raise KeyError(f"No object named '{name}' registerd in {self._name}.")

        return ret

    def hasRegistered(self, name):
        return name.upper() in self._obj_map
