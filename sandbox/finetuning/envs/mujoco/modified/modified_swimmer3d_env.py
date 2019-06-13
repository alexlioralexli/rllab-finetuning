import numpy as np
from rllab.core.serializable import Serializable
from sandbox.finetuning.envs.mujoco.swimmer3d_env import Swimmer3DEnv
from rllab.misc import logger
import os

class ModifiedSwimmer3DEnv(Swimmer3DEnv, Serializable):

    # mass: 4x1
    # dof_damping: 8x1
    # inertia: 4x3
    # geom_friction: 4x3

    def __init__(self, param_name=None, multiplier=np.ones((6,1)), *args, **kwargs):
        Serializable.quick_init(self, locals())
        super().__init__(*args, **kwargs)
        # import ipdb; ipdb.set_trace()
        if param_name is not None:
            multiplier = np.array(multiplier)
            assert param_name in ['None', 'body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
            assert hasattr(self.model, param_name)
            original_val = getattr(self.model, param_name, None)
            assert multiplier.shape == original_val.shape
            new_params = {param_name: original_val*multiplier}
            self.set_params(new_params)

    def set_params(self, new_params):
        for param, param_val in new_params.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)


if __name__ == '__main__':
    env = ModifiedSwimmer3DEnv(ego_obs=True)
    import ipdb; ipdb.set_trace()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()