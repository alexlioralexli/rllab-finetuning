import gym
from rllab.envs.mujoco.gather.gather_env import GatherEnv

class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

gym.spaces