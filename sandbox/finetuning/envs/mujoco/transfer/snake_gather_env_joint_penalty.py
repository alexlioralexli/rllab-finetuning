from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv
import time
from rllab.envs.base import Step
import numpy as np
from rllab.misc import logger

APPLE = 0
BOMB = 1


class SnakeGatherEnvJointPenalty(GatherEnv):
    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

    def penalty(self, joint_angles):
        threshold = 2.1
        coeff = 0.0001
        bias = 0.00005
        return np.sum((joint_angles[joint_angles > threshold] - threshold)*coeff + bias )

    def step(self, action):
        # import ipdb; ipdb.set_trace()
        _, inner_rew, done, info = self.wrapped_env.step(action)
        info['inner_rew'] = inner_rew

        joint_angles = np.absolute(self.wrapped_env.model.data.qpos[2:, 0])
        penalty = self.penalty(joint_angles)
        info['outer_rew'] = - penalty
        info['penalty'] = penalty
        info['joint_angles'] = joint_angles
        if done:
            return Step(self.get_current_obs(), self.dying_cost - penalty, done,
                        **info)  # give a -10 rew if the robot dies
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        reward = self.coef_inner_rew * inner_rew - penalty

        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                    info['outer_rew'] = 1 - penalty
                else:
                    reward = reward - 1
                    info['outer_rew'] = -1 - penalty
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0

        return Step(self.get_current_obs(), reward, done, **info)

    def log_diagnostics(self, paths, log_prefix='Gather', *args, **kwargs):
        penalty_sum = sum([path['env_infos']['penalty'].sum() for path in paths])
        with logger.tabular_prefix(log_prefix):
            logger.record_tabular("Penalty", penalty_sum)

if __name__ == "__main__":
    env = SnakeGatherEnvJointPenalty()
    import ipdb;

    ipdb.set_trace()
    while True:
        env.reset()
        for _ in range(1000):
            #            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
