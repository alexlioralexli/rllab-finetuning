from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv
import time
from rllab.envs.base import Step
import numpy as np
from rllab.misc import logger

APPLE = 0
BOMB = 1


class SnakeGatherEnvComPenalty(GatherEnv):
    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

    def penalty(self, com):
        threshold = 0.111
        coeff = 0.0005
        bias = 0.001
        if com > threshold:
            return bias + coeff * (com - threshold)
        else:
            return 0

    def step(self, action):
        # import ipdb; ipdb.set_trace()
        _, inner_rew, done, info = self.wrapped_env.step(action)
        info['inner_rew'] = inner_rew

        com_velocity = self.wrapped_env.get_body_comvel('torso')
        penalty = self.penalty(np.linalg.norm(com_velocity))
        info['outer_rew'] = - penalty
        info['penalty'] = penalty
        info['com_velocity_value'] = np.linalg.norm(com_velocity)
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
    env = SnakeGatherEnvComPenalty()
    import ipdb;

    ipdb.set_trace()
    while True:
        env.reset()
        for _ in range(1000):
            #            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
