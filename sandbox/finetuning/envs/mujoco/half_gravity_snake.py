from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv
from rllab.core.serializable import Serializable
from rllab.misc import autoargs


class HalfGravitySnakeEnv(SnakeEnv):
    FILE = 'snake.xml'
    ORI_IND = 2

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            ego_obs=False,
            sparse_rew=False,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.ego_obs = ego_obs
        self.sparse_rew = sparse_rew
        super(SnakeEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.model.opt.gravity[2] = -9.81/2
