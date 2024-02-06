# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import importlib
from collections import OrderedDict

from minigrid.envs.babyai.core.levelgen import LevelGen

from langsuite.envs.babyai import bonus_levels
from langsuite.envs.babyai import test_levels
RegisteredEnvList = [
    # https://github.com/Farama-Foundation/Minigrid.git
    # BabyAI - Language based levels - GoTo
    # ----------------------------------------
    dict(
        id="BabyAI-GoToRedBallGrey-v0",
        entry_point="minigrid.envs.babyai:GoToRedBallGrey",
    ),
    dict(
        id="BabyAI-GoToRedBall-v0",
        entry_point="minigrid.envs.babyai:GoToRedBall",
    ),
    dict(
        id="BabyAI-GoToRedBallNoDists-v0",
        entry_point="minigrid.envs.babyai:GoToRedBallNoDists",
    ),
    dict(
        id="BabyAI-GoToObj-v0",
        entry_point="minigrid.envs.babyai:GoToObj",
    ),
    dict(
        id="BabyAI-GoToObjS4-v0",
        entry_point="minigrid.envs.babyai:GoToObj",
        kwargs={"room_size": 4},
    ),
    dict(
        id="BabyAI-GoToObjS6-v1",
        entry_point="minigrid.envs.babyai:GoToObj",
        kwargs={"room_size": 6},
    ),
    dict(
        id="BabyAI-GoToLocal-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
    ),
    dict(
        id="BabyAI-GoToLocalS5N2-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 5, "num_dists": 2},
    ),
    dict(
        id="BabyAI-GoToLocalS6N2-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 6, "num_dists": 2},
    ),
    dict(
        id="BabyAI-GoToLocalS6N3-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 6, "num_dists": 3},
    ),
    dict(
        id="BabyAI-GoToLocalS6N4-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 6, "num_dists": 4},
    ),
    dict(
        id="BabyAI-GoToLocalS7N4-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 7, "num_dists": 4},
    ),
    dict(
        id="BabyAI-GoToLocalS7N5-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 7, "num_dists": 5},
    ),
    dict(
        id="BabyAI-GoToLocalS8N2-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 2},
    ),
    dict(
        id="BabyAI-GoToLocalS8N3-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 3},
    ),
    dict(
        id="BabyAI-GoToLocalS8N4-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 4},
    ),
    dict(
        id="BabyAI-GoToLocalS8N5-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 5},
    ),
    dict(
        id="BabyAI-GoToLocalS8N6-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 6},
    ),
    dict(
        id="BabyAI-GoToLocalS8N7-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 7},
    ),
    dict(
        id="BabyAI-GoTo-v0",
        entry_point="minigrid.envs.babyai:GoTo",
    ),
    dict(
        id="BabyAI-GoToOpen-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"doors_open": True},
    ),
    dict(
        id="BabyAI-GoToObjMaze-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "doors_open": False},
    ),
    dict(
        id="BabyAI-GoToObjMazeOpen-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "doors_open": True},
    ),
    dict(
        id="BabyAI-GoToObjMazeS4R2-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 4, "num_rows": 2, "num_cols": 2},
    ),
    dict(
        id="BabyAI-GoToObjMazeS4-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 4},
    ),
    dict(
        id="BabyAI-GoToObjMazeS5-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 5},
    ),
    dict(
        id="BabyAI-GoToObjMazeS6-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 6},
    ),
    dict(
        id="BabyAI-GoToObjMazeS7-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 7},
    ),
    dict(
        id="BabyAI-GoToImpUnlock-v0",
        entry_point="minigrid.envs.babyai:GoToImpUnlock",
    ),
    dict(
        id="BabyAI-GoToSeq-v0",
        entry_point="minigrid.envs.babyai:GoToSeq",
    ),
    dict(
        id="BabyAI-GoToSeqS5R2-v0",
        entry_point="minigrid.envs.babyai:GoToSeq",
        kwargs={"room_size": 5, "num_rows": 2, "num_cols": 2, "num_dists": 4},
    ),
    dict(
        id="BabyAI-GoToRedBlueBall-v0",
        entry_point="minigrid.envs.babyai:GoToRedBlueBall",
    ),
    dict(
        id="BabyAI-GoToDoor-v0",
        entry_point="minigrid.envs.babyai:GoToDoor",
    ),
    dict(
        id="BabyAI-GoToObjDoor-v0",
        entry_point="minigrid.envs.babyai:GoToObjDoor",
    ),
    # BabyAI - Language based levels - Open
    # ----------------------------------------
    dict(
        id="BabyAI-Open-v0",
        entry_point="minigrid.envs.babyai:Open",
    ),
    dict(
        id="BabyAI-OpenRedDoor-v0",
        entry_point="minigrid.envs.babyai:OpenRedDoor",
    ),
    dict(
        id="BabyAI-OpenDoor-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
    ),
    dict(
        id="BabyAI-OpenDoorDebug-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
        kwargs={"debug": True, "select_by": None},
    ),
    dict(
        id="BabyAI-OpenDoorColor-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
        kwargs={"select_by": "color"},
    ),
    dict(
        id="BabyAI-OpenDoorLoc-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
        kwargs={"select_by": "loc"},
    ),
    dict(
        id="BabyAI-OpenTwoDoors-v0",
        entry_point="minigrid.envs.babyai:OpenTwoDoors",
    ),
    dict(
        id="BabyAI-OpenRedBlueDoors-v0",
        entry_point="minigrid.envs.babyai:OpenTwoDoors",
        kwargs={"first_color": "red", "second_color": "blue"},
    ),
    dict(
        id="BabyAI-OpenRedBlueDoorsDebug-v0",
        entry_point="minigrid.envs.babyai:OpenTwoDoors",
        kwargs={
            "first_color": "red",
            "second_color": "blue",
            "strict": True,
        },
    ),
    dict(
        id="BabyAI-OpenDoorsOrderN2-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"num_doors": 2},
    ),
    dict(
        id="BabyAI-OpenDoorsOrderN4-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"num_doors": 4},
    ),
    dict(
        id="BabyAI-OpenDoorsOrderN2Debug-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"debug": True, "num_doors": 2},
    ),
    dict(
        id="BabyAI-OpenDoorsOrderN4Debug-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"debug": True, "num_doors": 4},
    ),
    # BabyAI - Language based levels - Pickup
    # ----------------------------------------
    dict(
        id="BabyAI-Pickup-v0",
        entry_point="minigrid.envs.babyai:Pickup",
    ),
    dict(
        id="BabyAI-UnblockPickup-v0",
        entry_point="minigrid.envs.babyai:UnblockPickup",
    ),
    dict(
        id="BabyAI-PickupLoc-v0",
        entry_point="minigrid.envs.babyai:PickupLoc",
    ),
    dict(
        id="BabyAI-PickupDist-v0",
        entry_point="minigrid.envs.babyai:PickupDist",
    ),
    dict(
        id="BabyAI-PickupDistDebug-v0",
        entry_point="minigrid.envs.babyai:PickupDist",
        kwargs={"debug": True},
    ),
    dict(
        id="BabyAI-PickupAbove-v0",
        entry_point="minigrid.envs.babyai:PickupAbove",
    ),
    # BabyAI - Language based levels - PutNext
    # ----------------------------------------
    dict(
        id="BabyAI-PutNextLocal-v0",
        entry_point="minigrid.envs.babyai:PutNextLocal",
    ),
    dict(
        id="BabyAI-PutNextLocalS5N3-v0",
        entry_point="minigrid.envs.babyai:PutNextLocal",
        kwargs={"room_size": 5, "num_objs": 3},
    ),
    dict(
        id="BabyAI-PutNextLocalS6N4-v0",
        entry_point="minigrid.envs.babyai:PutNextLocal",
        kwargs={"room_size": 6, "num_objs": 4},
    ),
    dict(
        id="BabyAI-PutNextS4N1-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 4, "objs_per_room": 1},
    ),
    dict(
        id="BabyAI-PutNextS5N2-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 5, "objs_per_room": 2},
    ),
    dict(
        id="BabyAI-PutNextS5N1-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 5, "objs_per_room": 1},
    ),
    dict(
        id="BabyAI-PutNextS6N3-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 6, "objs_per_room": 3},
    ),
    dict(
        id="BabyAI-PutNextS7N4-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 7, "objs_per_room": 4},
    ),
    dict(
        id="BabyAI-PutNextS5N2Carrying-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 5, "objs_per_room": 2, "start_carrying": True},
    ),
    dict(
        id="BabyAI-PutNextS6N3Carrying-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 6, "objs_per_room": 3, "start_carrying": True},
    ),
    dict(
        id="BabyAI-PutNextS7N4Carrying-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 7, "objs_per_room": 4, "start_carrying": True},
    ),
    # BabyAI - Language based levels - Unlock
    # ----------------------------------------
    dict(
        id="BabyAI-Unlock-v0",
        entry_point="minigrid.envs.babyai:Unlock",
    ),
    dict(
        id="BabyAI-UnlockLocal-v0",
        entry_point="minigrid.envs.babyai:UnlockLocal",
    ),
    dict(
        id="BabyAI-UnlockLocalDist-v0",
        entry_point="minigrid.envs.babyai:UnlockLocal",
        kwargs={"distractors": True},
    ),
    dict(
        id="BabyAI-KeyInBox-v0",
        entry_point="minigrid.envs.babyai:KeyInBox",
    ),
    dict(
        id="BabyAI-UnlockPickup-v0",
        entry_point="minigrid.envs.babyai:UnlockPickup",
    ),
    dict(
        id="BabyAI-UnlockPickupDist-v0",
        entry_point="minigrid.envs.babyai:UnlockPickup",
        kwargs={"distractors": True},
    ),
    dict(
        id="BabyAI-BlockedUnlockPickup-v0",
        entry_point="minigrid.envs.babyai:BlockedUnlockPickup",
    ),
    dict(
        id="BabyAI-UnlockToUnlock-v0",
        entry_point="minigrid.envs.babyai:UnlockToUnlock",
    ),
    # BabyAI - Language based levels - Other
    # ----------------------------------------
    dict(
        id="BabyAI-ActionObjDoor-v0",
        entry_point="minigrid.envs.babyai:ActionObjDoor",
    ),
    dict(
        id="BabyAI-FindObjS5-v0",
        entry_point="minigrid.envs.babyai:FindObjS5",
    ),
    dict(
        id="BabyAI-FindObjS6-v0",
        entry_point="minigrid.envs.babyai:FindObjS5",
        kwargs={"room_size": 6},
    ),
    dict(
        id="BabyAI-FindObjS7-v0",
        entry_point="minigrid.envs.babyai:FindObjS5",
        kwargs={"room_size": 7},
    ),
    dict(
        id="BabyAI-KeyCorridor-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
    ),
    dict(
        id="BabyAI-KeyCorridorS3R1-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 3, "num_rows": 1},
    ),
    dict(
        id="BabyAI-KeyCorridorS3R2-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 3, "num_rows": 2},
    ),
    dict(
        id="BabyAI-KeyCorridorS3R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 3, "num_rows": 3},
    ),
    dict(
        id="BabyAI-KeyCorridorS4R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 4, "num_rows": 3},
    ),
    dict(
        id="BabyAI-KeyCorridorS5R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 5, "num_rows": 3},
    ),
    dict(
        id="BabyAI-KeyCorridorS6R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 6, "num_rows": 3},
    ),
    dict(
        id="BabyAI-OneRoomS8-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
    ),
    dict(
        id="BabyAI-OneRoomS12-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
        kwargs={"room_size": 12},
    ),
    dict(
        id="BabyAI-OneRoomS16-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
        kwargs={"room_size": 16},
    ),
    dict(
        id="BabyAI-OneRoomS20-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
        kwargs={"room_size": 20},
    ),
    dict(
        id="BabyAI-MoveTwoAcrossS5N2-v0",
        entry_point="minigrid.envs.babyai:MoveTwoAcross",
        kwargs={"room_size": 5, "objs_per_room": 2},
    ),
    dict(
        id="BabyAI-MoveTwoAcrossS8N9-v0",
        entry_point="minigrid.envs.babyai:MoveTwoAcross",
        kwargs={"room_size": 8, "objs_per_room": 9},
    ),
    # BabyAI - Language based levels - Synth
    # ----------------------------------------
    dict(
        id="BabyAI-Synth-v0",
        entry_point="minigrid.envs.babyai:Synth",
    ),
    dict(
        id="BabyAI-SynthS5R2-v0",
        entry_point="minigrid.envs.babyai:Synth",
        kwargs={"room_size": 5, "num_rows": 2},
    ),
    dict(
        id="BabyAI-SynthLoc-v0",
        entry_point="minigrid.envs.babyai:SynthLoc",
    ),
    dict(
        id="BabyAI-SynthSeq-v0",
        entry_point="minigrid.envs.babyai:SynthSeq",
    ),
    dict(
        id="BabyAI-MiniBossLevel-v0",
        entry_point="minigrid.envs.babyai:MiniBossLevel",
    ),
    dict(
        id="BabyAI-BossLevel-v0",
        entry_point="minigrid.envs.babyai:BossLevel",
    ),
    dict(
        id="BabyAI-BossLevelNoUnlock-v0",
        entry_point="minigrid.envs.babyai:BossLevelNoUnlock",
    ),
]


# Dictionary of levels, indexed by name, lexically sorted
level_dict = OrderedDict()


def register_levels(env_list):
    """
    Register OpenAI gym environments for all levels in a file
    """
    # Iterate through global names
    for env in env_list:
        try:
            module_name, class_name = env["entry_point"].split(":")
            level_class = getattr(importlib.import_module(module_name), class_name)
            kwargs = env.get("kwargs", dict())
            level_name = env["id"].split("-")[1]
            if not level_class or isinstance(level_class, LevelGen):
                continue

            # Add the level to the dictionary
            level_dict[level_name] = level_class

            # Store the name and gym id on the level class
            level_class.level_name = level_name
            level_class.gym_id = env["id"]
            level_class.kwargs = kwargs

        except ImportError:
            continue


# Register the levels found in this file
# Compatible with https://github.com/mila-iqia/babyai/blob/master/babyai/levels/levelgen.py
register_levels(RegisteredEnvList)
register_levels(bonus_levels.RegisteredEnvList)
register_levels(test_levels.RegisteredEnvList)
