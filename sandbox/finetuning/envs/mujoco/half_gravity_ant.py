from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
from sandbox.finetuning.envs.mujoco.mujoco_env import MujocoEnv_ObsInit as MujocoEnv

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math
from sandbox.finetuning.envs.mujoco.ant_env import AntEnv


class HalfGravityAntEnv(AntEnv):
    FILE = 'low_gear_ratio_ant.xml'
    ORI_IND = 3

    def __init__(self,
                 ctrl_cost_coeff=1e-2,  # gym has 1 here!
                 rew_speed=False,  # if True the dot product is taken with the speed instead of the position
                 rew_dir=None,  # (x,y,z) -> Rew=dot product of the CoM SPEED with this dir. Otherwise, DIST to 0
                 ego_obs=False,
                 no_contact=False,
                 sparse=False,
                 *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.reward_dir = rew_dir
        self.rew_speed = rew_speed
        self.ego_obs = ego_obs
        self.no_cntct = no_contact
        self.sparse = sparse

        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.model.opt.gravity[2] = -9.81*0.9

if __name__ == "__main__":
    env = HalfGravityAntEnv()
    import IPython; IPython.embed()