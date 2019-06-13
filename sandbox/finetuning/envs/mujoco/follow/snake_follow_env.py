from sandbox.finetuning.envs.mujoco.follow.follow_env import FollowEnv
from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv


class SnakeFollowEnv(FollowEnv):
    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

