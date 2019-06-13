from rllab.core.serializable import Serializable
import numpy as np
from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv
from rllab.misc import autoargs

BIG = 1e6


class SnakeBasinEnv(SnakeEnv):
    FILE = 'snake_hfield.xml'
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
        # for the hfield
        self.height_walls = np.array([3, -3, 3, 0, -3., 3, -3]) * -1.
        # self.height_walls = np.array([3, 3, -3, -3, 3, 3, 0, -3., -3, 3, 3, -3, -3]) * -1.
        self.height = 0.8
        self.width = 10
        height = self.height
        width = self.width
        self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 55
        # self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
        self.model.hfield_size = np.array([50, 10, height, 0.1])
        row = np.zeros((500,))
        for i, x in enumerate(self.x_walls):
            terrain = np.cumsum([self.height_walls[i]] * width)
            row[x:x + width] += terrain
            row[x + width:] = row[x + width - 1]
        row = (row - np.min(row)) / (np.max(row) - np.min(row))
        hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
        # hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
        self.model.hfield_data = hfield