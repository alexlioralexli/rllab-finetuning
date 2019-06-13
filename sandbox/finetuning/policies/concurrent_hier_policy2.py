from rllab.policies.base import StochasticPolicy
from sandbox.finetuning.policies.test2_hier_snn_mlp_policy import GaussianMLPPolicy_snn_hier
from rllab.envs.normalized_env import normalize
from sandbox.finetuning.envs.mujoco.swimmer_env import SwimmerEnv
from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from sandbox.finetuning.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.finetuning.envs.hierarchized_snn_env import hierarchize_snn
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.core.lasagne_powered import LasagnePowered
from rllab import spaces
from rllab.envs.env_spec import EnvSpec
from rllab.misc import logger
import math
import numpy as np
import joblib
import os
from rllab.misc.tensor_utils import flatten_tensors
from scipy.special import entr
from rllab import config



# todo: check the lasagne powered, look at the output layers in the init method
# class HierarchicalPolicy(StochasticPolicy, LasagnePowered, Serializable):
class HierarchicalPolicy(StochasticPolicy, LasagnePowered, Parameterized, Serializable):
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
            pkl_path=None, # for the entire hierarchical policy, can take in npz too!
            snn_pkl_path=None,  # can actually be either pkl or npz
            snn_json_path=None,
            manager_pkl_path=None,  # default is to initialize a new manager from scratch
            period=2,  # how often the manager chooses latent skill
            latent_dim=6,
            bilinear_integration=True,
            trainable_snn=True,
            trainable_manager=True,
            continuous_latent=False,
            hidden_sizes_snn=(64, 64),
            hidden_sizes_selector=(32, 32)):
        StochasticPolicy.__init__(self, env_spec)
        self.env = env
        self.period = period
        self.latent_dim = latent_dim  # unsure
        self.bilinear_integration = bilinear_integration  # unsure
        self.count = 0  # keep track of how long it's been since sampling a latent skill
        self.curr_latent = None  # something
        self.outer_action_space = spaces.Discrete(latent_dim)
        self.trainable_manager = trainable_manager
        self.continuous_latent = continuous_latent
        self.trainable_snn = trainable_snn
        if pkl_path and '.npz' not in pkl_path:
            data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
            policy = data['policy']
            self.manager = policy.manager
            self.low_policy = policy.low_policy

            #following two lines used for random manager
            # outer_env_spec = EnvSpec(observation_space=self.env.observation_space, action_space=self.outer_action_space)
            # self.manager = CategoricalMLPPolicy(env_spec=outer_env_spec, latent_dim=latent_dim, )
        else:
            if snn_pkl_path is not None and '.npz' in snn_pkl_path:
                npz_path = snn_pkl_path
                snn_pkl_path = None
            else:
                npz_path = None

            self.low_policy = GaussianMLPPolicy_snn_hier(
                env_spec=env.spec,
                env=env,
                pkl_path=snn_pkl_path,
                npz_path=npz_path,
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
                    outer_env_spec = EnvSpec(observation_space=self.env.observation_space,
                                             action_space=spaces.Box(-1.0, 1.0, shape=(latent_dim,)))
                    self.manager = GaussianMLPPolicy(env_spec=outer_env_spec, learn_std=False)
                else:
                    outer_env_spec = EnvSpec(observation_space=self.env.observation_space,
                                             action_space=self.outer_action_space)
                    self.manager = CategoricalMLPPolicy(env_spec=outer_env_spec, latent_dim=latent_dim, )
                # import ipdb; ipdb.set_trace()
                if pkl_path is not None and '.npz' in pkl_path:
                    param_dict = dict(np.load(os.path.join(config.PROJECT_PATH, pkl_path)))
                    param_values = param_dict['params']
                    self.set_param_values(param_values)

        Serializable.quick_init(self, locals()) # todo: is this where this belongs?

    def get_action(self, observation):
        time_remaining = (self.period - self.count) / self.period
        if self.count % self.period == 0:  # sample a new latent skill
            latent, manager_dist_info = self.manager.get_action(observation)
            if self.continuous_latent:
                self.curr_latent = manager_dist_info['mean']  #i think the shape should be right now
                manager_dist_info = dict()
            else:
                self.curr_latent = self.outer_action_space.flatten(latent)  # make change here for skill removal
            # print("latent", self.curr_latent)

            self.count = (self.count + 1) % self.period
            action, skill_dist_info = self.low_policy.get_action((observation, self.curr_latent))
            skill_dist_info.update(manager_dist_info)
            skill_dist_info['time_remaining'] = time_remaining
            return action, skill_dist_info
        else:
            self.count = (self.count + 1) % self.period
            action, skill_dist_info = self.low_policy.get_action((observation, self.curr_latent))
            skill_dist_info['prob'] = np.array([])
            skill_dist_info['time_remaining'] = time_remaining
            return action, skill_dist_info

    def reset(self):
        self.count = 0

    def log_diagnostics(self, paths):
        # timesteps = 0
        # manager_entropy = 0.0
        # # skill_entropies = [0.0 for _ in range(self.latent_dim)]
        # skill_entropy = 0.0

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
        if not self.continuous_latent:
            manager_probs = np.zeros([1, self.latent_dim])
            for path in paths:
                latents = path['agent_infos']['latents']
                manager_probs += np.sum(latents, axis=0)
            manager_probs = manager_probs.reshape(-1)
            manager_probs = manager_probs / np.sum(manager_probs)
            logger.record_tabular("ManagerEntropy", entr(manager_probs).sum())
            logger.record_tabular("ManagerProbs", manager_probs)

    def distribution(self):
        raise NotImplementedError

    def dist_info_sym(self, obs_var, state_info_vars):
        if self.latent_dim == 1:
            return self.low_policy.dist_info_sym(obs_var, state_info_vars)
        else:
            raise NotImplementedError

    def dist_info(self, obs, state_infos):
        raise NotImplementedError

    # not sure if this actually works
    def get_params(self, **tags):
        # skill_params = self.low_policy.get_params_snn()
        skill_params = self.low_policy.get_params(**tags)
        manager_params = self.manager.get_params(**tags)
        return skill_params + manager_params

    def set_param_values(self, flattened_params, **tags):
        # import pdb; pdb.set_trace()
        # temp = self.low_policy.get_params(**tags)
        # index = sum([np.prod(temp[i].shape.eval()) for i in range(len(temp))])
        index = len(self.low_policy.get_param_values(**tags))
        # index = len(flatten(temp))
        # index = len(self.low_policy.get_params_snn())  # todo: make a more efficient way to calculate this
        # self.low_policy.set_params_snn(flattened_params[:index])
        if self.trainable_snn and False: # not going to do std reinit
            self.low_policy.set_param_values_transfer(flattened_params[:index], **tags)
        else:
            self.low_policy.set_param_values(flattened_params[:index], **tags)
        self.manager.set_param_values(flattened_params[index:], **tags)
