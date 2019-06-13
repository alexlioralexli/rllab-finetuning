from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.disabled_ant_env import DisabledAntEnv


class DisabledAntGatherEnv(GatherEnv):

    MODEL_CLASS = DisabledAntEnv
    ORI_IND = 3

