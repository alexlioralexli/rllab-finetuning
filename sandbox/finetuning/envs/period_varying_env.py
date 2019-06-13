from rllab.envs.base import Env
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import numpy as np

class PeriodVaryingEnv(Env, Serializable):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env

        low, high = np.min(self._wrapped_env.spec.observation_space.low), np.max(
            self._wrapped_env.spec.observation_space.high)
        # assert len(wrapped_env.spec.observation_space.shape) == 1
        shape = (self._wrapped_env.spec.observation_space.shape[0] + 1,)
        self.obs_space = Box(low, high, shape)


    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        # return self._wrapped_env.observation_space
        return self.obs_space
        # initialized the observation space
        # low, high = np.min(self._wrapped_env.spec.observation_space.low), np.max(self._wrapped_env.spec.observation_space.high)
        # assert len(wrapped_env.spec.observation_space.shape) == 1
        # shape = (self._wrapped_env.spec.observation_space.shape[0] + 1,)
        # return Box(low, high, shape)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)



if __name__ == "__main__":
    from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
    env = SwimmerGatherEnv()
    import IPython; IPython.embed()