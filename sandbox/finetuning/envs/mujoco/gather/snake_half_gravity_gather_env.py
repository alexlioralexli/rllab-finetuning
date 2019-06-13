from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.half_gravity_snake import HalfGravitySnakeEnv
import math

class HalfGravitySnakeGatherEnv(GatherEnv):

    MODEL_CLASS = HalfGravitySnakeEnv
    ORI_IND = 2

if __name__ == "__main__":
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    env = HalfGravitySnakeGatherEnv(**keyword_args)
    import ipdb; ipdb.set_trace()