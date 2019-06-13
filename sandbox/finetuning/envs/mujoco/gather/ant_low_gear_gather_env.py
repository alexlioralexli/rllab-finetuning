from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.ant_env import AntEnv
import time

class AntLowGearGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 3

if __name__ == "__main__":
    env = AntLowGearGatherEnv()
    import ipdb; ipdb.set_trace()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
