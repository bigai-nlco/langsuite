# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import os


def load_assets(asset_path):
    all_objects = dict()
    if asset_path is None or not os.path.exists(asset_path):
        raise FileNotFoundError(f"asset path {asset_path} does not exist")

    with open(asset_path, "r", encoding="utf-8") as f:
        assets = json.load(f)
        for a in assets:
            for o in assets[a]:
                property = dict()
                property["bbox"] = o["boundingBox"]
                # print(o['objectType'], property["bbox"])
                property["objectType"] = o["objectType"]
                if "primaryProperty" in o.keys():
                    property["primaryProperty"] = o["primaryProperty"]
                else:
                    property["primaryProperty"] = None
                if "secondaryProperties" in o.keys():
                    property["secondaryProperties"] = o["secondaryProperties"]
                else:
                    property["primaryProperty"] = None
                all_objects[o["assetId"]] = property
    return all_objects


def load_object_metadata(metadata_path):
    metadata = dict()
    if metadata_path is None or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata path {metadata_path} does not exist")

    with open(metadata_path, "r", encoding="utf-8") as f:
        obj_meta = json.load(f)
        for a in obj_meta:
            # print(a)
            for o in obj_meta[a]:
                # print(len(o))
                for p in o:
                    if p["assetId"] != "":
                        property = p
                        del property["axisAlignedBoundingBox"]
                        metadata[p["assetId"]] = property
        return metadata


def load_receptacles(receptacles_path):
    receptacles = dict()
    if receptacles_path is None or not os.path.exists(receptacles_path):
        raise FileNotFoundError(f"receptacles path {receptacles_path} does not exist")

    with open(receptacles_path, "r", encoding="utf-8") as f:
        receptacles_dict = json.load(f)
        for a in receptacles_dict:
            receptacles[a] = []
            for o in receptacles_dict[a]:
                receptacles[a].append(o)
