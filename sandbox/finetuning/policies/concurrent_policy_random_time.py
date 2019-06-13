from rllab.policies.base import StochasticPolicy
from sandbox.finetuning.policies.test_hier_snn_mlp_policy import GaussianMLPPolicy_snn_hier
from rllab.envs.normalized_env import normalize
from sandbox.finetuning.envs.mujoco.swimmer_env import SwimmerEnv
from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from sandbox.finetuning.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.core.lasagne_powered import LasagnePowered
from rllab import spaces
from rllab.envs.env_spec import EnvSpec
import math
import numpy as np
import joblib
import os
from contextlib import contextmanager

from rllab import config
from sandbox.finetuning.policies.concurrent_hier_policy import HierarchicalPolicy
from sandbox.finetuning.envs.period_varying_env import PeriodVaryingEnv
from rllab.envs.mujoco.gather.gather_env import GatherEnv
from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.normalized_env import NormalizedEnv  # this is just to check if the env passed is a normalized maze


# todo: check the lasagne powered, look at the output layers in the init method
# class HierarchicalPolicy(StochasticPolicy, LasagnePowered, Serializable):
class HierarchicalPolicyRandomTime(HierarchicalPolicy):
    """
    This class is built to contain the entire hierarchical policy,
    both the manager and the skills network, so that it can interact with a normal environment


    Concern: may need to pull out all the internal workings from GaussianMLPPolicy_snn_hier,
    CategoricalMLPPolicy, in order to do the gradients correctly

    """

    def __init__(
            self,
            env_spec,
            env,  # the inner one, I believe
            pkl_path=None,  # for the entire hierarchical policy
            snn_pkl_path=None,
            snn_json_path=None,
            manager_pkl_path=None,  # default is to initialize a new manager from scratch
            max_period=10,  # possible periods
            latent_dim=6,
            bilinear_integration=True,
            trainable_snn=True,
            trainable_manager=True,
            hidden_sizes_snn=(64, 64),
            hidden_sizes_selector=(32, 32)):
        StochasticPolicy.__init__(self, env_spec)
        self.env = env
        self.periods = np.arange(1, max_period + 1)
        assert len(self.periods) > 0
        self.curr_period = self.periods[0]
        self.max_period = max(self.periods)
        self.latent_dim = latent_dim  # unsure
        self.bilinear_integration = bilinear_integration  # unsure
        self.count = 0  # keep track of how long it's been since sampling a latent skill
        self.curr_latent = None  # something
        self.outer_action_space = spaces.Discrete(latent_dim)
        self.trainable_manager = trainable_manager
        self.random_period = True
        self.fake_env = PeriodVaryingEnv(env)

        if pkl_path:
            data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
            policy = data['policy']
            self.manager = policy.manager
            self.low_policy = policy.low_policy

            # following two lines used for random manager
            # outer_env_spec = EnvSpec(observation_space=self.env.observation_space, action_space=self.outer_action_space)
            # self.manager = CategoricalMLPPolicy(env_spec=outer_env_spec, latent_dim=latent_dim, )
        else:
            # env spec that includes the extra parameter for time
            self.low_policy = GaussianMLPPolicy_snn_hier(
                env_spec=self.fake_env.spec,
                env=self.fake_env,
                pkl_path=snn_pkl_path,
                json_path=snn_json_path,
                trainable_snn=trainable_snn,
                latent_dim=latent_dim,
                bilinear_integration=bilinear_integration,
                external_latent=True,
                hidden_sizes_snn=hidden_sizes_snn,
                hidden_sizes_selector=hidden_sizes_selector
            )

            # loading manager from pkl file
            if manager_pkl_path:
                manager_data = joblib.load(os.path.join(config.PROJECT_PATH, manager_pkl_path))
                self.manager = manager_data['policy']
                print("loaded manager")
            else:
                # self.outer_env = hierarchize_snn(self.env, time_steps_agg=10, pkl_path=snn_pkl_path)
                outer_env_spec = EnvSpec(observation_space=self.fake_env.observation_space,
                                         action_space=self.outer_action_space)
                self.manager = CategoricalMLPPolicy(env_spec=outer_env_spec, latent_dim=latent_dim, )

        if isinstance(env, MazeEnv) or isinstance(env, GatherEnv):
            self.obs_robot_dim = env.robot_observation_space.flat_dim
            self.obs_maze_dim = env.maze_observation_space.flat_dim
        elif isinstance(env, NormalizedEnv):
            if isinstance(env.wrapped_env, MazeEnv) or isinstance(env.wrapped_env, GatherEnv):
                self.obs_robot_dim = env.wrapped_env.robot_observation_space.flat_dim
                self.obs_maze_dim = env.wrapped_env.maze_observation_space.flat_dim
            else:
                self.obs_robot_dim = env.wrapped_env.observation_space.flat_dim
                self.obs_maze_dim = 0
        else:
            self.obs_robot_dim = env.observation_space.flat_dim
            self.obs_maze_dim = 0
        Serializable.quick_init(self, locals())  # todo: ask if this fixes my problem

    def get_random_period(self):
        return self.periods[np.random.choice(len(self.periods))]

    def get_action(self, observation):
        resampled = False
        time_remaining, extended_obs = None, None
        if self.count % self.curr_period == 0:  # sample a new latent skill
            if self.random_period:
                # print("Resampling, old period:", self.curr_period)
                self.curr_period = self.get_random_period()
                # print("New period:", self.curr_period)
            time_remaining = (self.curr_period - self.count) / self.max_period
            extended_obs = np.insert(observation, self.obs_robot_dim, time_remaining)
            self.curr_latent = self.outer_action_space.flatten(self.manager.get_action(extended_obs)[0])
            # print("latent", self.curr_latent)
            self.low_policy.set_pre_fix_latent(self.curr_latent)
            self.low_policy.reset()
            resampled = True
        if time_remaining is None or extended_obs is None:
            time_remaining = (self.curr_period - self.count) / self.max_period
            extended_obs = np.insert(observation, self.obs_robot_dim, time_remaining)
        # print("Time remaining,", time_remaining)
        action, info_dict = self.low_policy.get_action(extended_obs)
        info_dict['resampled_period'] = resampled
        info_dict['time_remaining'] = time_remaining
        self.count = (self.count + 1) % self.curr_period
        return action, info_dict

    @contextmanager
    def fix_period(self, period):
        prev_period, prev_random_period = self.curr_period, self.random_period
        self.curr_period = period
        self.random_period = False
        yield
        self.curr_period = prev_period
        self.random_period = prev_random_period

    def reset(self):
        self.count = 0

    def log_diagnostics(self, paths):
        # timesteps = 0
        # manager_entropy = 0.0
        # # skill_entropies = [0.0 for _ in range(self.latent_dim)]
        # skill_entropy = 0.0
        #
        # for path in paths:
        #     timesteps += len(path['observations'])
        #
        #     # calculate the entropy of the categorical distribution at each stage
        #     manager_dist_info = self.manager.dist_info(path['observations'])
        #     manager_entropy += self.manager.distribution.entropy(manager_dist_info).sum()
        #
        #     # calculate the entropy of each skill
        #     latent_dist_infos = self.low_policy.dist_info_sym_all_latents(path['observations'])
        #     # for i in range(len(latent_dist_infos)):
        #     #     latent_dist_info = {'log_std': latent_dist_infos[i]['log_std'].eval()}
        #     #     skill_entropies[i] += self.low_policy.distribution.entropy(latent_dist_info).sum()
        #     latent_dist_info = {'log_std': latent_dist_infos[0]['log_std'].eval()}
        #     skill_entropy += self.low_policy.distribution.entropy(latent_dist_info).sum()
        #
        # logger.record_tabular("AverageManagerEntropy", manager_entropy/timesteps)
        # # for i in range(self.latent_dim):
        # #     logger.record_tabular("AverageLatent{0}Entropy".format(str(i)), skill_entropies[i]/timesteps)
        # logger.record_tabular("AverageLatentEntropy", skill_entropy / timesteps)
        pass

    def distribution(self):
        raise NotImplementedError

    def dist_info_sym(self, obs_var, state_info_vars):
        if self.latent_dim == 1:
            return self.low_policy.dist_info_sym(obs_var, state_info_vars)
        else:
            raise NotImplementedError

