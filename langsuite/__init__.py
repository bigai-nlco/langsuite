import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from langsuite.envs.base_env import make_env
from langsuite.task import make
