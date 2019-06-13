import numpy as np
from rllab.core.serializable import Serializable
from sandbox.finetuning.envs.mujoco.mujoco_env import MujocoEnv_ObsInit as MujocoEnv
from rllab.misc import logger


class AntEnv(MujocoEnv, Serializable):
    FILE = 'low_gear_ratio_ant.xml'
    ORI_IND = 3

    def __init__(self, ego_obs=False, no_contact=False, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.cripple_mask = None
        self.ego_obs = ego_obs
        self.no_cntct = no_contact
        MujocoEnv.__init__(self, *args, **kwargs)
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        self.task = 'cripple'
        self.crippled_leg = 0
        if "file_path" in kwargs: # this is being used in gather
            self.gather = True
        else:
            self.gather = False
        self.reset_task(value=self.crippled_leg)

    def get_current_obs(self):
        # return np.concatenate([
        #     self.model.data.qpos.flat,
        #     self.model.data.qvel.flat,
        #     self.get_body_xmat("torso").flat,
        #     self.get_body_com("torso"),
        # ]).reshape(-1)
        if self.ego_obs:
            return np.concatenate([
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
            ]).reshape(-1)
        elif self.no_cntct:
            return np.concatenate([
                self.model.data.qpos.flat,
                self.model.data.qvel.flat,
                self.get_body_xmat("torso").flat,
                self.get_body_com("torso"),
            ]).reshape(-1)
        else:
            return np.concatenate([
                self.model.data.qpos.flat,
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
                self.get_body_xmat("torso").flat,
                self.get_body_com("torso"),
            ]).reshape(-1)

    def step(self, action):
        action = self.cripple_mask * action
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        # lb, ub = self.action_space.low, self.action_space.high
        # scaling = (ub - lb) * 0.5
        ctrl_cost = 0  # 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self.get_current_obs()
        info = {}
        return ob, reward, done, info

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        # lb, ub = self.action_bounds
        # scaling = (ub - lb) * 0.5
        ctrl_cost = 0  # 5e-3 * np.sum(np.square(action / scaling), axis=1)
        vel = (next_obs[:, -3] - obs[:, -3]) / self.dt
        survive_reward = 0.05
        reward = vel - ctrl_cost + survive_reward
        return reward


    '''
    our "front" is in +x direction, to the right side of screen

    LEG 4 (they call this back R)
    action0: front-right leg, top joint 
    action1: front-right leg, bottom joint

    LEG 1 (they call this front L)
    action2: front-left leg, top joint
    action3: front-left leg, bottom joint 

    LEG 2 (they call this front R)
    action4: back-left leg, top joint
    action5: back-left leg, bottom joint 

    LEG 3 (they call this back L)
    action6: back-right leg, top joint
    action7: back-right leg, bottom joint 

    geom_names has 
            ['floor','torso_geom',
            'aux_1_geom','left_leg_geom','left_ankle_geom', --1
            'aux_2_geom','right_leg_geom','right_ankle_geom', --2
            'aux_3_geom','back_leg_geom','third_ankle_geom', --3
            'aux_4_geom','rightback_leg_geom','fourth_ankle_geom'] --4
    '''

    def reset_task(self, value=None):

        if self.task == 'cripple':
            # Pick which leg to remove (0 1 2 are train... 3 is test)
            self.crippled_leg = value if value is not None else np.random.randint(0, 3)

            # Pick which actuators to disable
            self.cripple_mask = np.ones(self.action_space.shape)
            if self.crippled_leg == 0:
                self.cripple_mask[2] = 0
                self.cripple_mask[3] = 0
            elif self.crippled_leg == 1:
                self.cripple_mask[4] = 0
                self.cripple_mask[5] = 0
            elif self.crippled_leg == 2:
                self.cripple_mask[6] = 0
                self.cripple_mask[7] = 0
            elif self.crippled_leg == 3:
                self.cripple_mask[0] = 0
                self.cripple_mask[1] = 0

            if self.gather:
                offset = 4
            else:
                offset = 0

            # Make the removed leg look red
            geom_rgba = self._init_geom_rgba.copy()
            if self.crippled_leg == 0:
                geom_rgba[3+offset, :3] = np.array([1, 0, 0])
                geom_rgba[4+offset, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 1:
                geom_rgba[6+offset, :3] = np.array([1, 0, 0])
                geom_rgba[7+offset, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 2:
                geom_rgba[9+offset, :3] = np.array([1, 0, 0])
                geom_rgba[10+offset, :3] = np.array([1, 0, 0])
            elif self.crippled_leg == 3:
                geom_rgba[12+offset, :3] = np.array([1, 0, 0])
                geom_rgba[13+offset, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

            # Make the removed leg not affect anything
            temp_size = self._init_geom_size.copy()
            temp_pos = self._init_geom_pos.copy()

            if self.crippled_leg == 0:
                # Top half
                temp_size[3+offset, 0] = temp_size[3+offset, 0] / 2
                temp_size[3+offset, 1] = temp_size[3+offset, 1] / 2
                # Bottom half
                temp_size[4+offset, 0] = temp_size[4+offset, 0] / 2
                temp_size[4+offset, 1] = temp_size[4+offset, 1] / 2
                temp_pos[4+offset, :] = temp_pos[3+offset, :]

            elif self.crippled_leg == 1:
                # Top half
                temp_size[6+offset, 0] = temp_size[6+offset, 0] / 2
                temp_size[6+offset, 1] = temp_size[6+offset, 1] / 2
                # Bottom half
                temp_size[7+offset, 0] = temp_size[7+offset, 0] / 2
                temp_size[7+offset, 1] = temp_size[7+offset, 1] / 2
                temp_pos[7+offset, :] = temp_pos[6+offset, :]

            elif self.crippled_leg == 2:
                # Top half
                temp_size[9+offset, 0] = temp_size[9+offset, 0] / 2
                temp_size[9+offset, 1] = temp_size[9+offset, 1] / 2
                # Bottom half
                temp_size[10+offset, 0] = temp_size[10+offset, 0] / 2
                temp_size[10+offset, 1] = temp_size[10+offset, 1] / 2
                temp_pos[10+offset, :] = temp_pos[9+offset, :]

            elif self.crippled_leg == 3:
                # Top half
                temp_size[12+offset, 0] = temp_size[12+offset, 0] / 2
                temp_size[12+offset, 1] = temp_size[12+offset, 1] / 2
                # Bottom half
                temp_size[13+offset, 0] = temp_size[13+offset, 0] / 2
                temp_size[13+offset, 1] = temp_size[13+offset, 1] / 2
                temp_pos[13+offset, :] = temp_pos[12+offset, :]

            self.model.geom_size = temp_size
            self.model.geom_pos = temp_pos

        else:
            raise NotImplementedError

        self.model.forward()

    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
        logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
        logger.logkv(prefix + 'StdForwardProgress', np.std(progs))


if __name__ == '__main__':
    env = AntEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()