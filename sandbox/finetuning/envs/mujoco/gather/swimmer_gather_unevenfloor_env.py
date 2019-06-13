from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.swimmer_unevenfloor_env import SwimmerUnevenFloorEnv

class SwimmerGatherUnevenFloorEnv(GatherEnv):

    MODEL_CLASS = SwimmerUnevenFloorEnv
    ORI_IND = 2


if __name__ == "__main__":
    env = SwimmerGatherUnevenFloorEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
    # env.reset()
    # frames = []
    # for i in range(5):
    #     print(i)
    #     frames.append(env.render(mode='rgb_array'))
    #     _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
    #
    # import skvideo.io
    # import numpy as np
    # output_data = np.array(frames)
    # import IPython; IPython.embed()
    # output_data = output_data.astype(np.uint8)
    # import os.path as osp
    # import rllab.config as config
    #
    # output_path = osp.join(config.PROJECT_PATH, "data/local/outputvideo.mp4")
    # skvideo.io.vwrite(output_path, output_data)