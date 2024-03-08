from __future__ import annotations

import re


def camelcase(string):
    """Convert string into camel case."""
    string = re.sub(r"\w[\s\W]+\w", "", str(string))
    if not string:
        return string
    return (string[0]).lower() + re.sub(
        r"[\-_\.\s]([a-z])", lambda matched: (matched.group(1)).upper(), string[1:]
    )
