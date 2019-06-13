from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.base import Step

APPLE = 0
BOMB = 1

class SwimmerGatherReversedEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

    def step(self, action):
        _, inner_rew, done, info = self.wrapped_env.step(action)
        info['inner_rew'] = inner_rew
        info['outer_rew'] = 0
        if done:
            return Step(self.get_current_obs(), self.dying_cost, done, **info)  # give a -10 rew if the robot dies
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        reward = self.coef_inner_rew * inner_rew
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward -1
                    info['outer_rew'] = -1
                else:
                    reward = reward + 1
                    info['outer_rew'] = +1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return Step(self.get_current_obs(), reward, done, **info)
