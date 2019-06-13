from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.modified.snake_hfield_env import SnakeHfieldEnv

class SnakeHfieldGather(GatherEnv):
    MODEL_CLASS = SnakeHfieldEnv
    ORI_IND = 2


if __name__ == "__main__":
    env = SnakeHfieldGather()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action