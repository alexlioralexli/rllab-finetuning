from sandbox.finetuning.envs.mujoco.ant_env import AntEnv

class DisabledAntEnv(AntEnv):
    """
    This AntEnv differs from the one in rllab.envs.mujoco.ant_env in the additional initialization options
    that fix the rewards and observation space. Sparse strips the forward reward and keeps ctrl, contact, survival
    The 'com' is added to the env_infos.
    Also the get_ori() method is added for the Maze and Gather tasks.
    AND we kill with z<0.3!!!! (not 0.2 as in gym)
    """
    FILE = 'low_gear_ratio_ant.xml'
    ORI_IND = 3

    def step(self, action):
        action = action.copy()
        action[0] = 0.0
        return super().step(action)