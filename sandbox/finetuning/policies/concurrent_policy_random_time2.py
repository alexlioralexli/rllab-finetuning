from rllab.policies.base import StochasticPolicy
from sandbox.finetuning.policies.test2_hier_snn_mlp_policy import GaussianMLPPolicy_snn_hier
from sandbox.finetuning.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.core.serializable import Serializable
from rllab import spaces
from rllab.envs.env_spec import EnvSpec
import math
import numpy as np
import joblib
import os
from contextlib import contextmanager
from rllab.misc import logger
from rllab import config
from scipy.special import entr
from sandbox.finetuning.policies.concurrent_hier_policy2 import HierarchicalPolicy
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
            min_period=1,
            max_period=10,  # possible periods
            latent_dim=6,
            bilinear_integration=True,
            trainable_snn=True,
            trainable_manager=True,
            continuous_latent=False,
            hidden_sizes_snn=(64, 64),
            hidden_sizes_selector=(32, 32)):
        StochasticPolicy.__init__(self, env_spec)
        self.env = env
        self.periods = np.arange(min_period, max_period + 1)
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

        self.continuous_latent = continuous_latent
        self.trainable_snn = trainable_snn

        if pkl_path and '.npz' not in pkl_path:
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
                if self.continuous_latent:
                    outer_env_spec = EnvSpec(observation_space=self.fake_env.observation_space,
                                             action_space=spaces.Box(-1.0, 1.0, shape=(latent_dim,)))
                    self.manager = GaussianMLPPolicy(env_spec=outer_env_spec)
                else:
                    outer_env_spec = EnvSpec(observation_space=self.fake_env.observation_space,
                                             action_space=self.outer_action_space)
                    self.manager = CategoricalMLPPolicy(env_spec=outer_env_spec, latent_dim=latent_dim, )
                if pkl_path is not None and '.npz' in pkl_path:
                    param_dict = dict(np.load(os.path.join(config.PROJECT_PATH, pkl_path)))
                    param_values = param_dict['params']
                    self.set_param_values(param_values)

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
        if self.count % self.curr_period == 0:  # sample a new latent skill
            if self.random_period:
                # print("Resampling, old period:", self.curr_period)
                self.curr_period = self.get_random_period()
                # print("New period:", self.curr_period)
            time_remaining = (self.curr_period - self.count) / self.max_period
            extended_obs = np.insert(observation, self.obs_robot_dim, time_remaining)
            latent, info_dict = self.manager.get_action(extended_obs)
            if self.continuous_latent:
                self.curr_latent = latent[0]
            else:
                self.curr_latent = self.outer_action_space.flatten(latent)
            # print("latent", self.curr_latent)
            info_dict['resampled_period'] = True
        else:
            time_remaining = (self.curr_period - self.count) / self.max_period
            extended_obs = np.insert(observation, self.obs_robot_dim, time_remaining)
            info_dict = dict(prob=np.zeros(self.latent_dim), resampled_period=False)
        # print("Time remaining,", time_remaining)
        action, skill_dist_info = self.low_policy.get_action((extended_obs, self.curr_latent))
        info_dict.update(skill_dist_info)
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
        manager_probs = np.zeros([1, self.latent_dim])
        for path in paths:
            latents = path['agent_infos']['latents']
            manager_probs += np.sum(latents, axis=0)
        manager_probs = manager_probs.reshape(-1)
        manager_probs = manager_probs / np.sum(manager_probs)
        logger.record_tabular("ManagerEntropy", entr(manager_probs).sum())
        logger.record_tabular("ManagerProbs", manager_probs)