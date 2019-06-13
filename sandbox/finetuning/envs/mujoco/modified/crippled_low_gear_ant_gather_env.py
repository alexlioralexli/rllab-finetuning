from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.modified.crippled_low_gear_ant_env import AntEnv

class CrippledAntLowGearGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 3

if __name__ == "__main__":
    env = CrippledAntLowGearGatherEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action