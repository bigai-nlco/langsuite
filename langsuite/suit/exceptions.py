from typing import Dict, NewType

from overrides import EnforceOverrides

class StructuredException(Exception, EnforceOverrides):
    def __init__(self, param_dict, *args) -> None:
        super().__init__(*args)
        self.param_dict: Dict[str, object] = param_dict


class NotRegisteredError(StructuredException):
    """TODO"""


class ParameterMissingError(StructuredException):
    """Some of the parameters are not provided. e.g., action name, object name"""


class IllegalActionError(StructuredException):
    """The parameter are illegal in the current state. e.g., can not found the object in vision"""


class InvalidActionError(StructuredException):
    """No need to exec."""


class LimitExceededError(StructuredException):
    """Numeric value exceeded certain limit."""


class UnexecutableWithSptialError(StructuredException):
    """Some of the premise on sptial relations not satisfied."""

class UnexecutableWithAttrError(StructuredException):
    """Some of the premise on object's attributes not satisfied."""
