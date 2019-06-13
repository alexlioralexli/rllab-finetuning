from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.snake_basin_env import SnakeBasinEnv
import math

class SnakeBasinGatherEnv(GatherEnv):

    MODEL_CLASS = SnakeBasinEnv
    ORI_IND = 2

if __name__ == "__main__":
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    env = SnakeBasinGatherEnv(**keyword_args)
    import ipdb; ipdb.set_trace()