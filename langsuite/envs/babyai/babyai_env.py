# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from gymnasium import spaces

from langsuite.agents import Agent
from langsuite.envs.babyai.levels import RegisteredEnvList, level_dict
from langsuite.envs.base_env import ENV_REGISTRY, LangSuiteEnv
from langsuite.utils.logging import logger

BabyAIAction = {
    "BabyAITurnLeft": 0,
    "BabyAITurnRight": 1,
    "BabyAIMoveAhead": 2,
    "BabyAIPickUp": 3,
    "BabyAIDrop": 4,
    "BabyAIToggle": 5,
    "done": 6,
    "BabyAIStart": 7,
}


@ENV_REGISTRY.register()
class BabyAIEnv(LangSuiteEnv):
    registered_env_ids = [env["id"] for env in RegisteredEnvList]

    def __init__(self, env_config):
        super().__init__(env_config=env_config)
        self.agents = defaultdict()
        self.agent_ids = []
        # self.feedback_builder = TemplateBuilder(env_config.get("template"))
        self.action_spaces = spaces.Discrete(len(BabyAIAction))
        self.closed = False

    def create_world(self, world_cfg) -> None:
        cfg = deepcopy(world_cfg)
        world_type = cfg.pop("type")
        assert world_type == "BabyAIWorld", f"Invalid world type: {world_type}"
        world_id = cfg.pop("id")
        seed = cfg.pop("seed", 0)
        #HACK XXX
        if 'log_file' in world_cfg:
            setattr(self, 'task_log_file_name', world_cfg['log_file'])
        if world_id.startswith("BabyAI"):
            world_id = world_id.split("-")[1]
        if world_id in level_dict:
            level_kwargs = level_dict[world_id].kwargs
            logger.info(level_kwargs)
            self.world = level_dict[world_id](**level_kwargs)

        if self.world:
            self.world.seed = seed
            self.world.reset(seed=seed)

    def add_agent(self, agent_cfg) -> None:
        # raise
        if len(self.agents) == 0:
            if "position" in agent_cfg and type(agent_cfg.get("position")) == str:
                position = self.random_world_position()
                agent_cfg.update({"position": position})
                logger.info(agent_cfg)
            agent = Agent.create(agent_cfg)
            agent.set_env(self)
            agent.set_name(agent_cfg.get("name", "BabyAI"))
            agent.reset()
            self.agents[agent.id] = agent
            self.agent_ids.append(agent.id)
        else:
            logger.error("BabyAI only need one agent!")
            raise

    def is_valid_action(self, action: str) -> bool:
        if action in BabyAIAction:
            return True
        else:
            return False

    def update_config(self, config):
        for i, agent_id in enumerate(self.agents):
            self.agents[agent_id].update_config(config["agents"][i])

    # def get_task_info(self):
    #     return {
    #         "state": ActionFeedback(
    #             success=True, feedback=self.feedback_builder.build("intro")
    #         ).to_json()
    #     }

    def step(self, action_dict):
        """Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info."""
        next_action = None
        if len(self.agents.keys()) > 0:
            for id in self.agents.keys():
                agent_id = id
                break
            status, parsed_response = self.agents[agent_id].step(action_dict)
            # action_taken = BabyAIAction.get(parsed_response.get("action"))
            # task_success = self.world.instrs.verify(action_taken)
            # if action_taken:
            #     next_action = self.bot.replan(action_taken=action_taken)
            #     print(next_action)
            info = {
                "status": status,
                "agent": agent_id,
                "feedback": parsed_response["feedback"],
                "action": parsed_response.get("action"),
                "response": parsed_response.get("response"),
                "is_successfull": parsed_response.get("is_successfull"),
            }
        else:
            logger.info(f"Agent {agent_id} not found in environment.")
            info = {
                "status": False,
                "agent": agent_id,
                "feedback": f"Agent {agent_id} not found in environment.",
            }

        obs = self.world.gen_obs()
        # TODO transfer obs to text obs
        agent_pos = self.world.agent_pos
        crrt_cell = self.world.grid.get(*agent_pos)
        done = False
        reward = 0
        is_terminated = False
        # is_successfull = False
        # if task_success and task_success == "success":
        #     is_terminated = True
        #     is_successfull = True
        #     reward = 1
        #     done = True
        # if next_action and next_action == self.world.actions.done:
        #     is_terminated = True
        #     is_successfull = True
        #     reward = 1

        if "BabyAIStop" == parsed_response.get("action"):
            is_terminated = True
        if parsed_response.get("is_successfull", False):
            done = True
        info.update({"is_terminated": is_terminated})
        return obs, reward, done, info

    def reset(self):
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""
        self.world.reset()

    def render(self, mode="", **kwargs):
        """Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        grid, vis_mask = self.world.gen_obs_grid()
        figure = grid.render(
            16,
            agent_pos=(self.world.agent_view_size // 2, self.world.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )
        if mode == "webui":
            return figure
        return figure

    def close(self):
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""
        if self.world.window:
            self.world.window.close()
        return

    def get_feedback_builder(self):
        return self.feedback_builder

    def set_feedback_builder(self, feedback_builder):
        self.feedback_builder = feedback_builder

    def gen_obs(self, agent=None):
        if not agent:
            agent = list(self.agents.values())[0]

        grid, vis_mask = self.world.gen_obs_grid()
        f_vec = self.world.dir_vec
        r_vec = self.world.right_vec

        # Compute the absolute coordinates of the top-left corner
        # of the agent's view area
        top_left = (
            self.world.agent_pos
            + f_vec * (agent.view_size - 1)
            - r_vec * (agent.view_size // 2)
        )

        obs_mid = []
        obs_left = []
        obs_right = []
        # Mark everything in front of us as visible
        for vis_j in range(0, agent.view_size):
            for vis_i in range(0, agent.view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= agent.vis_mask.shape[0]:
                    continue
                if abs_j < 0 or abs_j >= agent.vis_mask.shape[1]:
                    continue

                agent.vis_mask[abs_i, abs_j] = True
                crrt_cell = self.world.grid.get(abs_i, abs_j)

                if crrt_cell and crrt_cell.type != "wall":
                    if vis_i < agent.view_size // 2:
                        obs_left.append(f"a {crrt_cell.color} {crrt_cell.type}")
                    elif vis_i == agent.view_size // 2:
                        obs_mid.append(f"a {crrt_cell.color} {crrt_cell.type}")
                    else:
                        obs_right.append(f"a {crrt_cell.color} {crrt_cell.type}")

        return {"left": obs_left, "mid": obs_mid, "right": obs_right}

    def get_observation(self, agent=None):
        obs = self.gen_obs(agent=agent)
        front_pos = self.world.front_pos
        front_cell = self.world.grid.get(*front_pos)
        manipulable_object = ""
        if front_cell and front_cell.type != "wall":
            manipulable_object += (
                "\nManipulable object: a "
                + front_cell.color
                + " "
                + front_cell.type
                + ";"
            )
        mid = obs["mid"]
        left = obs["left"]
        right = obs["right"]
        if len(mid) > 0:
            obs_mid = "You can see " + ", ".join(mid) + " in front of you; "
        else:
            obs_mid = ""

        if len(left) > 0:
            obs_left = "You can see " + ",".join(left) + " on your left; "
        else:
            obs_left = ""

        if len(right) > 0:
            obs_right = "You can see " + ",".join(right) + " on your right; "
        else:
            obs_right = ""

        all_obs = obs_mid + obs_left + obs_right + manipulable_object

        if len(all_obs) == 0:
            return "You can see nothing ahead."
        else:
            all_obs = all_obs.strip("; ")
            all_obs += "."
        return all_obs

    def get_held_object_observation(self):
        observation = "You are now holding "
        if self.world.carrying:
            observation += (
                "a " + self.world.carrying.color + " " + self.world.carrying.type + "."
            )
        else:
            observation += "nothing. "
        return observation

    def get_task_def(self):
        return self.world.mission
