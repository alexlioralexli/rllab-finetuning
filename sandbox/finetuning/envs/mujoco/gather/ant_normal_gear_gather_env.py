from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.ant_normal_gear_env import AntNormalGearEnv


class AntNormalGearGatherEnv(GatherEnv):

    MODEL_CLASS = AntNormalGearEnv
    ORI_IND = 3

if __name__ == "__main__":
    env = AntNormalGearGatherEnv()
    # import IPython; IPython.embed()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action