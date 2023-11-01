from __future__ import annotations

from langsuite.actions import simple_actions
from langsuite.actions.base_action import ACTION_REGISTERY, ActionFeedback


def get_action(action_name: str, env, agent, **kwargs):
    if not action_name:
        return None

    if "_" in action_name:
        action_name = "".join(
            [a[0].upper() + a[1:].lower() for a in action_name.split("_")]
        )

    if "Action" not in action_name:
        action_name = action_name + "Action"

    if not ACTION_REGISTERY.hasRegistered(action_name):
        return None

    return ACTION_REGISTERY.get(action_name)(env=env, agent=agent, **kwargs)
