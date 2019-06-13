import theano
import theano.tensor as TT
from sandbox.finetuning.policies.concurrent_hier_policy2 import HierarchicalPolicy
from sandbox.finetuning.policies.concurrent_policy_random_time2 import HierarchicalPolicyRandomTime
from sandbox.finetuning.algos.hier_batch_polopt import BatchPolopt, BatchSampler  # note that I use my own BatchPolopt class here
from sandbox.finetuning.algos.hier_batch_sampler import HierBatchSampler
from rllab.misc import ext
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import rllab.misc.logger as logger
import numpy as np
import copy
from rllab.spaces.box import Box
from rllab.envs.env_spec import EnvSpec
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.finetuning.algos.concurrent_ppo import Concurrent_PPO


# I use this for the mixture of p
class Hippo(BatchPolopt):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.0003,
                 latents=None,  # some sort of iterable of the actual latent vectors
                 average_period=10, # average over all the periods
                 truncate_local_is_ratio=None,
                 epsilon=0.1,
                 train_pi_iters=80,
                 use_skill_dependent_baseline=False,
                 mlp_skill_dependent_baseline=False,
                 **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                # optimizer_args = dict()
                optimizer_args = dict(batch_size=None)
            self.optimizer = FirstOrderOptimizer(learning_rate=step_size, max_epochs=train_pi_iters,
                                                 **optimizer_args)
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.epsilon = epsilon

        super(Hippo, self).__init__(**kwargs)  # not sure if this line is correct
        self.num_latents = kwargs['policy'].latent_dim
        self.latents = latents
        self.average_period = average_period

        # import pdb; pdb.set_trace()
        # self.sampler = BatchSampler(self)
        self.sampler = HierBatchSampler(self, period=None)

        # i hope this is right
        self.diagonal = DiagonalGaussian(self.policy.low_policy.action_space.flat_dim)
        self.debug_fns = []

        assert isinstance(self.policy, HierarchicalPolicyRandomTime)
        # self.old_policy = copy.deepcopy(self.policy)

        # skill dependent baseline
        self.use_skill_dependent_baseline = use_skill_dependent_baseline
        self.mlp_skill_dependent_baseline = mlp_skill_dependent_baseline
        if use_skill_dependent_baseline:
            curr_env = kwargs['env']
            skill_dependent_action_space = curr_env.action_space
            new_obs_space_no_bi = curr_env.observation_space.shape[0] + 1  # 1 for the t_remaining
            skill_dependent_obs_space_dim = (new_obs_space_no_bi * (self.num_latents + 1) + self.num_latents,)
            skill_dependent_obs_space = Box(-1.0, 1.0, shape=skill_dependent_obs_space_dim)
            skill_dependent_env_spec = EnvSpec(skill_dependent_obs_space, skill_dependent_action_space)
            if self.mlp_skill_dependent_baseline:
                self.skill_dependent_baseline = GaussianMLPBaseline(env_spec=skill_dependent_env_spec)
            else:
                self.skill_dependent_baseline = LinearFeatureBaseline(env_spec=skill_dependent_env_spec)

    def init_opt(self):
        obs_var = ext.new_tensor('obs', ndim=2, dtype=theano.config.floatX)  # todo: check the dtype

        manager_obs_var = ext.new_tensor('manager_obs', ndim=2, dtype=theano.config.floatX)

        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )

        # this will have to be the advantage every time the manager makes a decision
        manager_advantage_var = ext.new_tensor(
            'manager_advantage',
            ndim=1,
            dtype=theano.config.floatX
        )

        skill_advantage_var = ext.new_tensor(
            'skill_advantage',
            ndim=1,
            dtype=theano.config.floatX
        )

        latent_var_sparse = ext.new_tensor(
            'sparse_latent',
            ndim=2,
            dtype=theano.config.floatX
        )

        latent_var = ext.new_tensor(
            'latents',
            ndim=2,
            dtype=theano.config.floatX
        )

        mean_var = ext.new_tensor(
            'mean',
            ndim=2,
            dtype=theano.config.floatX
        )

        log_std_var = ext.new_tensor(
            'log_std',
            ndim=2,
            dtype=theano.config.floatX
        )

        manager_prob_var = ext.new_tensor(
            'log_std',
            ndim=2,
            dtype=theano.config.floatX
        )

        assert isinstance(self.policy, HierarchicalPolicy)

        #############################################################
        ### calculating the manager portion of the surrogate loss ###
        #############################################################

        # i, j should contain the probability of latent j at time step self.period*i
        # should be a len(obs)//self.period by len(self.latent) tensor
        latent_probs = self.policy.manager.dist_info_sym(manager_obs_var)['prob']
        # old_latent_probs = self.old_policy.manager.dist_info_sym(manager_obs_var)['prob']

        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)
        old_actual_latent_probs = TT.sum(manager_prob_var * latent_var_sparse, axis=1)
        lr = TT.exp(TT.log(actual_latent_probs) - TT.log(old_actual_latent_probs))
        manager_surr_loss_vector = TT.minimum(lr * manager_advantage_var,
                                              TT.clip(lr, 1 - self.epsilon, 1 + self.epsilon) * manager_advantage_var)
        manager_surr_loss = -TT.mean(manager_surr_loss_vector)

        ############################################################
        ### calculating the skills portion of the surrogate loss ###
        ############################################################

        dist_info_var = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
        old_dist_info_var = dict(mean=mean_var, log_std=log_std_var)
        skill_lr = self.diagonal.likelihood_ratio_sym(action_var, old_dist_info_var, dist_info_var)

        skill_surr_loss_vector = TT.minimum(skill_lr * skill_advantage_var,
                                            TT.clip(skill_lr, 1 - self.epsilon, 1 + self.epsilon) * skill_advantage_var)
        skill_surr_loss = -TT.mean(skill_surr_loss_vector)

        surr_loss = manager_surr_loss/self.average_period + skill_surr_loss

        input_list = [obs_var, manager_obs_var, action_var, manager_advantage_var, skill_advantage_var, latent_var,
                      latent_var_sparse, mean_var, log_std_var, manager_prob_var]

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            inputs=input_list
        )
        return dict()


    # do the optimization
    def optimize_policy(self, itr, samples_data):
        # print(len(samples_data['observations']), self.period)
        # assert len(samples_data['observations']) % self.period == 0

        # note that I have to do extra preprocessing to the advantages, and also create obs_var_sparse
        if self.use_skill_dependent_baseline:
            input_values = tuple(ext.extract(
                samples_data, "observations", "actions", "advantages", "agent_infos", "skill_advantages"))
        else:
            input_values = tuple(ext.extract(
                samples_data,
                "observations", "actions", "advantages", "agent_infos"
            ))

        time_remaining = input_values[3]['time_remaining']
        resampled_period = input_values[3]['resampled_period']
        obs_var = np.insert(input_values[0], self.policy.obs_robot_dim, time_remaining, axis=1)
        manager_obs_var = obs_var[resampled_period]
        action_var = input_values[1]
        manager_adv_var = input_values[2][resampled_period]

        latent_var = input_values[3]['latents']
        latent_var_sparse = latent_var[resampled_period]
        mean = input_values[3]['mean']
        log_std = input_values[3]['log_std']
        prob = input_values[3]['prob'][resampled_period]
        if self.use_skill_dependent_baseline:
            skill_adv_var = input_values[4]
            all_input_values = (obs_var, manager_obs_var, action_var, manager_adv_var,
                                skill_adv_var, latent_var, latent_var_sparse, mean, log_std, prob)
        else:
            skill_adv_var = input_values[2]
            all_input_values = (obs_var, manager_obs_var, action_var, manager_adv_var,
                            skill_adv_var, latent_var, latent_var_sparse, mean, log_std, prob)

        # todo: assign current parameters to old policy; does this work?
        # old_param_values = self.policy.get_param_values()
        # self.old_policy.set_param_values(old_param_values)
        loss_before = self.optimizer.loss(all_input_values)
        self.optimizer.optimize(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env
        )

    def log_diagnostics(self, paths):
        # paths obtained by self.sampler.obtain_samples
        BatchPolopt.log_diagnostics(self, paths)
        # self.sampler.log_diagnostics(paths)   # wasn't doing anything anyways

        # want to log the standard deviations
        # want to log the max and min of the actions