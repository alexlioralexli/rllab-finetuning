from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.modified.swimmer3d_hfield_env import Swimmer3DHfieldEnv


class Swimmer3dHfieldGather(GatherEnv):
    MODEL_CLASS = Swimmer3DHfieldEnv
    ORI_IND = 2


if __name__ == "__main__":
    env = Swimmer3dHfieldGather()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action