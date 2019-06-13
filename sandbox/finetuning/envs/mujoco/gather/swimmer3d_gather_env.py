from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.swimmer3d_env import Swimmer3DEnv
import math

class Swimmer3dGatherEnv(GatherEnv):
    MODEL_CLASS = Swimmer3DEnv
    ORI_IND = 2

if __name__ == "__main__":
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    env = Swimmer3dGatherEnv(**keyword_args)
    # import IPython; IPython.embed()
    while True:
        env.reset()
        for _ in range(1000):
            # env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action