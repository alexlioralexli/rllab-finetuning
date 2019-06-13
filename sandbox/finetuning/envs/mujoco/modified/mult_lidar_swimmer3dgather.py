import math

from rllab.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.finetuning.envs.mujoco.swimmer3d_env import Swimmer3DEnv
from rllab.core.serializable import Serializable

class MultSwimmer3dGatherEnv(GatherEnv, Serializable):

    MODEL_CLASS = Swimmer3DEnv
    ORI_IND = 2

    def __init__(
            self,
            multiplier=1.0,
            n_apples=8,
            n_bombs=8,
            activity_range=6.,
            robot_object_spacing=2.,
            catch_range=1.,
            n_bins=10,
            sensor_range=6.,
            sensor_span=math.pi,
            coef_inner_rew=0.,
            dying_cost=-10,
            *args, **kwargs
    ):
        Serializable.quick_init(self, locals())
        import ipdb; ipdb.set_trace()

        super().__init__(n_apples=8,
                        n_bombs=8,
                        activity_range=6.,
                        robot_object_spacing=2.,
                        catch_range=1.,
                        n_bins=10,
                        sensor_range=6.,
                        sensor_span=math.pi,
                        coef_inner_rew=0.,
                        dying_cost=-10,
                        *args, **kwargs)
        self.multiplier = multiplier

    def get_readings(self):
        return self.multiplier * super().get_readings()

if __name__ == "__main__":
    env = MultSwimmer3dGatherEnv(multiplier=2.0)
    while True:
        env.reset()
        for _ in range(1000):
            # env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action