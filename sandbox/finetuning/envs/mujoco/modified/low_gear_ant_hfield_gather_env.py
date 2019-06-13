from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.modified.low_gear_ant_hfield_env import AntHfieldEnv

class AntLowGearHfieldGatherEnv(GatherEnv):

    MODEL_CLASS = AntHfieldEnv
    ORI_IND = 3

if __name__ == "__main__":
    env = AntLowGearHfieldGatherEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action