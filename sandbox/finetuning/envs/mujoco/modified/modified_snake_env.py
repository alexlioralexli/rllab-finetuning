import numpy as np
from rllab.core.serializable import Serializable
from sandbox.finetuning.envs.mujoco.snake_env import SnakeEnv

class ModifiedSnakeEnv(SnakeEnv, Serializable):

    # mass: 6x1
    # dof_damping: 7x1
    # inertia: 6x3
    # geom_friction: 6x3

    def __init__(self, param_name=None, multiplier=np.ones((6,1)), *args, **kwargs):
        Serializable.quick_init(self, locals())
        super().__init__(*args, **kwargs)
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
    # env = ModifiedSnakeEnv()
    # env = ModifiedSnakeEnv(param_name="body_mass", multiplier=0.1*np.ones((6,1)))
    env = ModifiedSnakeEnv(param_name="geom_friction", multiplier=10*np.ones((6,3)))
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()