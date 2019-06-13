from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.half_gravity_ant import HalfGravityAntEnv


class HalfGravityAntGatherEnv(GatherEnv):

    MODEL_CLASS = HalfGravityAntEnv
    ORI_IND = 3

