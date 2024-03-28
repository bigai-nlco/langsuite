# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import json
import os
import random

from langsuite.utils.logging import logger


class TemplateBuilder:
    def __init__(self, template_json: str = None, template=None) -> None:
        if template is None and template_json is None:
            raise ValueError("One of 'template' and 'template_json' must be provied")

        if not template:
            if not os.path.exists(template_json) or not template_json.endswith(".json"):
                raise ValueError(f"Invalid template file {template_json}")
            with open(template_json, "r", encoding="utf-8") as jsonf:
                template = json.load(jsonf)

        self.template = template
        if not self._validate_template():
            raise ValueError("Invalid template.")

    def _validate_template(self):
        return True

    def build(self, domain: str, key: str = "default", *args, **kwargs):
        if domain not in self.template:
            logger.info(f"Invalid domain {domain}")
            return ""

        template = self.template.get(domain)
        if key not in template:
            logger.info(f"Key {key} not found in domain {domain}")
            return ""

        template = template.get(key)

        if type(template) == list:
            template = random.choice(template)
        for idx, arg in enumerate(args):
            template = template.replace("{" + str(idx) + "}", arg)

        for k, v in kwargs.items():
            template = template.replace("{" + str(k) + "}", str(v))

        return template
