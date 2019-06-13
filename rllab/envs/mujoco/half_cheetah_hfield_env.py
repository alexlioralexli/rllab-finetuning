import numpy as np
import os
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
# import logger


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahHFieldEnv(MujocoEnv, gym.utils.EzPickle):
    FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../vendor/mujoco_models/half_cheetah_hfield.xml')

    def __init__(self, task='hfield', reset_task_every_episode=False, *args, **kwargs):
        self.reset_task_every_episode = reset_task_every_episode
        self.first = True
        self._action_bounds = None
        MujocoEnv.__init__(self, self.FILE, 1)
        self._action_bounds = self.action_space.low, self.action_space.high
        gym.utils.EzPickle.__init__(self)

        # hfield default configuration
        self.x_walls = np.array([250, 260, 261, 270, 280, 285])
        self.height_walls = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.height = 0.8
        self.width = 15

        assert task in [None, 'None', 'hfield', 'same', 'hill', 'gentle', 'steep', 'basin']
        self.task = task

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def _step(self, action):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        next_obs = self.get_current_obs()
        forward_reward = (xposafter - xposbefore) / self.dt
        if self._action_bounds is not None:
            action = np.clip(action, *self._action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        reward = forward_reward - ctrl_cost
        done = False
        return next_obs, reward, done, dict()

    def get_reward(self, observation, next_observation, action):
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        vel = (next_observation[:, -3] - observation[:, -3])/self.dt
        reward = vel - ctrl_cost
        return reward

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.get_current_obs()
        if self.reset_task_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    def reset_task(self, value=None):
        if self.task == 'hfield':
            height = np.random.uniform(0.2, 1)
            width = 10
            n_walls = 6
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            x_walls = np.random.choice(np.arange(255, 310, width), replace=False, size=n_walls)
            x_walls.sort()
            sign = np.random.choice([1, -1], size=n_walls)
            sign[:2] = 1
            height_walls = np.random.uniform(0.2, 0.6, n_walls) * sign
            row = np.zeros((500,)) # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(x_walls):
                terrain = np.cumsum([height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x+width:] = row[x+width - 1]
            row = (row - np.min(row))/(np.max(row) - np.min(row))

            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)

            self.model.hfield_data = hfield


        elif self.task == 'same':
            height = self.height
            width = self.width
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'basin':
            ### TASK 1: BASIN ###
            self.height_walls = np.array([-1, 1, 0., 0., 0., 0.])  # basin
            self.height = 0.55
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'hill':
            ### TASK 2: HILL ###
            self.height_walls = np.array([1, -1, 0, 0., 0, 0])   # hill
            self.height  = 0.6
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield


        elif self.task == 'gentle':
            ### TASK 3: GENTLE SLOPE
            self.height_walls = np.array([1, 1, 1, 1, 1, 1]) # low slope
            self.height = 1
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'steep':
            ### TASK 4: STEEP SLOPE
            self.height_walls = np.array([1, 1, 1, 1, 1, 1]) # low slope
            self.height = 4
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'None' or self.task is None:
            pass
        else:
            raise NotImplementedError

        self.model.forward()

    def __getstate__(self):
        state = super(HalfCheetahHFieldEnv, self).__getstate__()
        state['task'] = self.task
        state['reset_task_every_episode'] = self.reset_task_every_episode
        return state

    def __setstate__(self, d):
        super(HalfCheetahHFieldEnv, self).__setstate__(d)
        self.task = d['task']
        self.reset_task_every_episode = d['reset_task_every_episode']

    def log_diagnostics(self, paths):
        """
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        logger.logkv('AverageForwardProgress', np.mean(progs))
        logger.logkv('MaxForwardProgress', np.max(progs))
        logger.logkv('MinForwardProgress', np.min(progs))
        logger.logkv('StdForwardProgress', np.std(progs))
        """
        pass


if __name__ == "__main__":
    # env = HalfCheetahHFieldEnv(None)
    env = HalfCheetahHFieldEnv(task="steep") #hill gentle steep
    # import IPython; IPython.embed()
    # print(os.path.realpath(__file__))
    # print(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../vendor/mujoco_models/half_cheetah_hfield.xml'))
    while True:
        task = env.reset_task()
        env.reset()
        for _ in range(1000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action

