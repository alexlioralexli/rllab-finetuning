import json
import os
from contextlib import contextmanager

import joblib
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano
import theano.tensor as TT

from rllab import config
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.envs.mujoco.gather.gather_env import GatherEnv
from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.normalized_env import NormalizedEnv  # this is just to check if the env passed is a normalized maze
from rllab.misc import autoargs
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy
from rllab.spaces import Box
from sandbox.finetuning.core.lasagne_layers import BilinearIntegrationLayer, CropLayer
from sandbox.finetuning.distributions.categorical import Categorical_oneAxis as Categorical
from rllab.distributions.bernoulli import Bernoulli


class ActionRepeatingPolicy(Policy, Serializable):  # also inherits from Parametrized

    def __init__(
            self,
            env_spec,
            env,
    ):
        Serializable.quick_init(self, locals())
        super(ActionRepeatingPolicy, self).__init__(env_spec)
        self.action = None

    @overrides
    def get_action(self, observation):
        assert self.action is not None
        # print("get action", self.action)
        return self.action, dict()

    def get_actions(self, observations):
        assert self.action is not None
        # print("get actions", self.action, len(observations))
        return np.repeat(np.reshape(self.action, (-1, 1)), len(observations), axis=1), dict()

    @contextmanager
    def fix_action(self, action):
        assert len(action.shape) == 1
        # print("set action", action)
        self.action = action
        yield
        self.action = None
        # print("unset action")