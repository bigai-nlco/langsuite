from typing import Any, Dict
from pettingzoo.utils.env import ParallelEnv

from langsuite.suit import AGENT_REGISTRY, LangSuiteAgent
from langsuite.suit import StructuredException
from langsuite.suit import World
from langsuite.utils import logging
from langsuite.suit import WORLD_REGISTRY

class LangSuiteEnv(ParallelEnv):
    """
    Env --START-> LangAgent --Action Decision-> World --Obs--> LangAgent -->

    :param ParallelEnv: _description_
    :type ParallelEnv: _type_
    """

    def __init__(self, env_data) -> None:
        # TODO add agents, world
        super().__init__()
        self.world = self.make_world(env_data)
        self.agents: Dict[str, LangSuiteAgent] = dict()
        self._terminated: Dict[str, bool] = dict()
        self._truncated: Dict[str, bool] = dict()
        self._step_count: Dict[str, int] = dict()
        self._rewards: Dict[str, Any] = dict()
        self.add_agents(env_data)

    @property
    def possible_agents(self):
        return [v for k, v in self.agents.items() if k not in self._terminated and k not in self._truncated]


    def make_world(self, env_data) -> World:
        # FIXME more world types
        world_type = env_data["world_type"]
        world_data = env_data["world_data"]
        world_data["agents"] = env_data["agents"]
        return WORLD_REGISTRY.get(world_type).create(world_data)
        # return Basic2DWorld.create_from_ProcThor(env_data['world_data'])

    def add_agents(self, env_data):
        for agent_data in env_data["agents"]:
            ag_type = agent_data["type"]
            ag_id = agent_data.get("agent_name", f"agent_{len(self.agents)}")
            agent = AGENT_REGISTRY.get(ag_type).create(
                name=ag_id, world=self.world, agent_data=agent_data
            )
            self.agents[ag_id] = agent
            self._terminated[ag_id] = False
            self._truncated[ag_id] = False
            self._rewards[ag_id] = 0
            self._step_count[ag_id] = 0

    def step(self, actions: dict):
        """Receives a dictionary of actions keyed by the agent name.

        Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary
        and info dictionary, where each dictionary is keyed by the agent.

        actions and feedback(info) are verbalized.
        """
        feedback_dict = {}
        # TODO check if agent stopped
        for agent_id, agent in self.agents.items():
            if agent_id in actions:
                try:
                    action_dict = agent.make_decision(actions[agent_id])
                    #Just a thought, load action.
                    if len(action_dict) == 0:
                        action_dict = agent.make_decision('OK.')
                    #Stop after decision, because it should know that it is stopped.
                    if not agent.stopped:
                        self._step_count[agent_id] += 1
                        _, feedback = self.world.step(agent_id, action_dict)
                        feedback_dict[agent_id] = feedback
                        if feedback_dict[agent_id]["action"] == "Stop":
                            agent.pre_stop()
                            self._rewards[agent_id] = feedback_dict[agent_id]['reward']
                    else:
                        logging.logger.info('%s stops', agent_id)
                        self._terminated[agent_id] = True
                except StructuredException as e:
                    feedback_dict[agent_id] = e.param_dict
                    feedback_dict[agent_id]["error_type"] = type(e)

        #TODO Theoratially, ParallelEnv should update after all executions, currently our update may simplifies it, we didn't force step to not update things.
        self.world.update()

        for agent_id, agent in self.agents.items():
            if agent_id in feedback_dict:
                #FIXME move this into agent and add hint info
                if self._step_count[agent_id] == agent.step_limit:
                    logging.logger.info('%s reach step limit %d', agent_id, agent.step_limit)
                    self._truncated[agent_id] = True

        observation_dict = {}
        for agent_id, agent in self.agents.items():
            if agent_id in feedback_dict:
                observation_dict[agent_id] = self.world.get_observation(agent_id)
                feedback_dict[agent_id] = agent.pack(
                    (feedback_dict[agent_id], observation_dict[agent_id])
                )
                logging.logger.debug("ag=%s, fb=%s", agent_id, feedback_dict[agent_id])

        # print(
        #     observation_dict,
        #     self._rewards,
        #     self._terminated,
        #     self._truncated,
        #     feedback_dict,
        # )

        return (
            observation_dict,
            self._rewards,
            self._terminated,
            self._truncated,
            feedback_dict,
        )
