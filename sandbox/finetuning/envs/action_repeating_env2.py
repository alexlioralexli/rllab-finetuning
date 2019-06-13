import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.normalized_env import NormalizedEnv
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.finetuning.envs.mujoco.maze.fast_maze_env import FastMazeEnv
from sandbox.finetuning.sampler.utils import rollout  # this is a different rollout (option of no reset)
from sandbox.finetuning.policies.action_repeating_policy import ActionRepeatingPolicy

"""
Wrapper environment that takes an action and uses it for the next time_steps_agg timesteps. Tests the most basic form of temporal abstraction.
"""

class ActionRepeatingEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            time_steps_agg=1,
            animate=False,
            keep_rendered_rgb=False,
    ):
        """
        :param env: Env to wrap, should have same robot characteristics than env where the policy where pre-trained on
        :param time_steps_agg: Time-steps during which the SNN policy is executed with fixed (discrete) latent
        :param keep_rendered_rgb: the returned frac_paths include all rgb images (for plotting video after)
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.time_steps_agg = time_steps_agg
        self.animate = animate
        self.keep_rendered_rgb = keep_rendered_rgb

    @overrides
    def step(self, action):
        i = 0
        done = False
        total_reward = 0
        while not done and i < self.time_steps_agg:
            i += 1
            obs, rew, done, info = self.wrapped_env.step(action)
            total_reward += rew
        return obs, total_reward, done, dict()

    def __str__(self):
        return "Action repeating env wrapped: %s" % self._wrapped_env

