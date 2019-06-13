from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.envs.mujoco.ant_hfield_env import AntHfieldEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
