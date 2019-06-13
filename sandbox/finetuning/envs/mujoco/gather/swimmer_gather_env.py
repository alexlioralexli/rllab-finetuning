from rllab.envs.mujoco.gather.gather_env import GatherEnv
# from rllab.envs.mujoco.gather.baselines_gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.swimmer_env import SwimmerEnv
import time

class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

if __name__ == "__main__":
    env = SwimmerGatherEnv()
    # import IPython; IPython.embed()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
