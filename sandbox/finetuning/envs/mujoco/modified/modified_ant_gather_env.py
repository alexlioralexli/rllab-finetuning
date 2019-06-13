from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.modified.modified_ant_env import ModifiedAntEnv

class ModifiedAntLowGearGatherEnv(GatherEnv):

    MODEL_CLASS = ModifiedAntEnv
    ORI_IND = 3

if __name__ == "__main__":
    env = ModifiedAntLowGearGatherEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action