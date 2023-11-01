from __future__ import annotations


class Expert:
    def __init__(self, expert_json) -> None:
        taskinfo = expert_json["task_info"]
        self.scene = taskinfo["scene"]
        self.index = taskinfo["index"]
        self.stage = taskinfo["stage"]
        self.actions = []
        self.agent_locs = []

        actions = taskinfo["unshuffle_actions"]
        agent_locs = taskinfo["agent_locs"]
        print(len(agent_locs))
        print(len(actions))

        action_successes = taskinfo["unshuffle_action_successes"]
        self.agent_locs.append(agent_locs[0])
        for i in range(len(actions)):
            if (
                actions[i].startswith("rotate")
                or actions[i].startswith("open")
                or actions[i].startswith("move")
                or actions[i].startswith("pickup")
                or actions[i].startswith("drop")
                or actions[i].startswith("done")
            ) and action_successes[i]:
                self.actions.append(actions[i])
                self.agent_locs.append(agent_locs[i + 1])
