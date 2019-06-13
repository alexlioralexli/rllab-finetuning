from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.modified.modified_snake_env import ModifiedSnakeEnv

class ModifiedSnakeGatherEnv(GatherEnv):

    MODEL_CLASS = ModifiedSnakeEnv
    ORI_IND = 2

if __name__ == "__main__":
    env = ModifiedSnakeGatherEnv()
    import ipdb; ipdb.set_trace()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action