from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv
import math

class SnakeGatherEnv(GatherEnv):

    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

if __name__ == "__main__":
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    env = SnakeGatherEnv(**keyword_args)
    import ipdb; ipdb.set_trace()