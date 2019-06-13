from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.modified.modified_swimmer3d_env import ModifiedSwimmer3DEnv

class ModifiedSwimmer3dGatherEnv(GatherEnv):

    MODEL_CLASS = ModifiedSwimmer3DEnv
    # MODEL_CLASS = AntEnv
    ORI_IND = 2

if __name__ == "__main__":
    env = ModifiedSwimmer3dGatherEnv()
    import ipdb; ipdb.set_trace()
    while True:
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action