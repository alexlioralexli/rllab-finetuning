import theano
import theano.tensor as TT
from sandbox.finetuning.policies.concurrent_hier_policy import HierarchicalPolicy
from sandbox.finetuning.algos.hier_batch_polopt import BatchPolopt, BatchSampler  # note that I use my own BatchPolopt class here
from rllab.misc import ext
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import rllab.misc.logger as logger
from sandbox.finetuning.algos.hier_batch_sampler import HierBatchSampler
import copy
from rllab.spaces.box import Box
from rllab.envs.env_spec import EnvSpec
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

class Concurrent_PPO(BatchPolopt):
    """
    Designed to enable concurrent training of a SNN that parameterizes skills
    and also train the manager at the same time

    Note that, if I'm not trying to do the sample approximation of the weird log of sum term,
    I don't need to know which skill was picked, just need to know the action
    """

    # double check this constructor later
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.0003,
                 num_latents=6,
                 latents=None,  # some sort of iterable of the actual latent vectors
                 period=10,  # how often I choose a latent
                 truncate_local_is_ratio=None,
                 epsilon=0.1,
                 train_pi_iters=80,
                 use_skill_dependent_baseline=False,
                 **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                # optimizer_args = dict()
                optimizer_args = dict(batch_size=None)
            self.optimizer = FirstOrderOptimizer(learning_rate=step_size, max_epochs=train_pi_iters, **optimizer_args)
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.epsilon = epsilon

        super(Concurrent_PPO, self).__init__(**kwargs)  # not sure if this line is correct
        self.num_latents = kwargs['policy'].latent_dim
        self.latents = latents
        self.period = period

        # todo: fix this sampler stuff
        # import pdb; pdb.set_trace()
        self.sampler = HierBatchSampler(self, self.period)
        # self.sampler = BatchSampler(self)
        # i hope this is right
        self.diagonal = DiagonalGaussian(self.policy.low_policy.action_space.flat_dim)
        self.debug_fns = []

        assert isinstance(self.policy, HierarchicalPolicy)
        if self.policy is not None:
            self.period = self.policy.period
        assert self.policy.period == self.period
        self.old_policy = copy.deepcopy(self.policy)

        # skill dependent baseline
        self.use_skill_dependent_baseline = use_skill_dependent_baseline
        if use_skill_dependent_baseline:
            curr_env = kwargs['env']
            skill_dependent_action_space = curr_env.action_space
            skill_dependent_obs_space_dim = ((curr_env.observation_space.shape[0] + 1) * self.num_latents,)
            skill_dependent_obs_space = Box(-1.0, 1.0, shape=skill_dependent_obs_space_dim)
            skill_depdendent_env_spec = EnvSpec(skill_dependent_obs_space, skill_dependent_action_space)
            self.skill_dependent_baseline = LinearFeatureBaseline(env_spec=skill_depdendent_env_spec)

    # initialize the computation graph
    # optimize is run on >= 1 trajectory at a time

    def init_opt(self):
        # obs_var_raw = self.env.observation_space.new_tensor_variable(
        #     'obs',
        #     extra_dims=1,
        # )


        obs_var_raw = ext.new_tensor('obs', ndim=3, dtype=theano.config.floatX)  # todo: check the dtype

        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )

        # this will have to be the advantage every self.period timesteps
        advantage_var_sparse = ext.new_tensor(
            'sparse_advantage',
            ndim=1,
            dtype=theano.config.floatX
        )

        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1,
            dtype=theano.config.floatX
        )

        obs_var_sparse = ext.new_tensor(
            'sparse_obs',
            ndim=2,
            dtype=theano.config.floatX  # todo: check this with carlos, refer to discrete.py in rllab.spaces
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

        assert isinstance(self.policy, HierarchicalPolicy)

        # todo: assumptions: 1 trajectory, which is a multiple of p; that the obs_var_probs is valid

        # undoing the reshape, so that batch sampling is ok
        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])
        # obs_var = obs_var_raw

        #############################################################
        ### calculating the manager portion of the surrogate loss ###
        #############################################################

        # i, j should contain the probability of latent j at time step self.period*i
        # should be a len(obs)//self.period by len(self.latent) tensor
        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
        old_latent_probs = self.old_policy.manager.dist_info_sym(obs_var_sparse)['prob']

        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)
        old_actual_latent_probs = TT.sum(old_latent_probs * latent_var_sparse, axis=1)
        lr = actual_latent_probs / old_actual_latent_probs
        manager_surr_loss_vector = TT.minimum(lr * advantage_var_sparse,
                                              TT.clip(lr, 1 - self.epsilon, 1 + self.epsilon) * advantage_var_sparse)
        manager_surr_loss = -TT.mean(manager_surr_loss_vector)
        # manager_surr_loss = - TT.mean(TT.log(actual_latent_probs) * advantage_var_sparse)

        ############################################################
        ### calculating the skills portion of the surrogate loss ###
        ############################################################

        # get the distribution parameters
        # dist_info_vars = []
        # for latent in self.latents:
        #     self.policy.low_policy.set_latent_train(latent)
        #     dist_info_vars.append(self.policy.low_policy.dist_info_sym(obs_var))
        # hopefully the above line takes multiple samples, and state_info_vars not needed as input

        dist_info_vars = self.policy.low_policy.dist_info_sym_all_latents(obs_var)
        probs = TT.stack([self.diagonal.log_likelihood_sym(action_var, dist_info) for dist_info in dist_info_vars],
                         axis=1)
        actual_action_log_probs = TT.sum(probs * latent_var, axis=1)  # todo: verify that dist_info_vars is in order

        # old policy stuff
        old_dist_info_vars = self.old_policy.low_policy.dist_info_sym_all_latents(obs_var)
        old_probs = TT.stack([self.diagonal.log_likelihood_sym(action_var, dist_info) for dist_info in old_dist_info_vars],
                        axis=1)
        old_actual_action_log_probs = TT.sum(old_probs * latent_var, axis=1)
        skill_lr = TT.exp(actual_action_log_probs - old_actual_action_log_probs)

        skill_surr_loss_vector = TT.minimum(skill_lr * advantage_var,
                                            TT.clip(skill_lr, 1 - self.epsilon, 1 + self.epsilon) * advantage_var)
        skill_surr_loss = -TT.mean(skill_surr_loss_vector)
        # skill_surr_loss = - TT.mean(actual_action_log_probs*advantage_var)

        surr_loss = manager_surr_loss/self.period + skill_surr_loss  # so that the relative magnitudes are correct

        input_list = [obs_var_raw, obs_var_sparse, action_var, advantage_var, advantage_var_sparse, latent_var,
                      latent_var_sparse]
        # input_list = [obs_var_raw, obs_var_sparse, action_var, advantage_var]
        # npo has state_info_vars and old_dist_info_vars, I don't think I need them until I go for NPO/TRPO

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            inputs=input_list
        )
        return dict()

    # do the optimization
    def optimize_policy(self, itr, samples_data):
        print(len(samples_data['observations']), self.period)
        assert len(samples_data['observations']) % self.period == 0

        # note that I have to do extra preprocessing to the advantages, and also create obs_var_sparse

        if self.use_skill_dependent_baseline:
            input_values = tuple(ext.extract(
                samples_data,
                "observations", "actions", "advantages", "agent_infos", "skill_advantages"
            ))
        else:
            input_values = tuple(ext.extract(
                samples_data,
                "observations", "actions", "advantages", "agent_infos"
            ))

        obs_raw = input_values[0].reshape(input_values[0].shape[0] // self.period, self.period,
                                          input_values[0].shape[1])

        obs_sparse = input_values[0].take([i for i in range(0, input_values[0].shape[0], self.period)], axis=0)
        advantage_sparse = input_values[2].reshape([input_values[2].shape[0] // self.period, self.period])[:, 0]
        latents = input_values[3]['latents']
        latents_sparse = latents.take([i for i in range(0, latents.shape[0], self.period)], axis=0)

        if self.use_skill_dependent_baseline:
            all_input_values = (
                obs_raw, obs_sparse, input_values[1], input_values[4], advantage_sparse, latents, latents_sparse)
        else:
            all_input_values = (
            obs_raw, obs_sparse, input_values[1], input_values[2], advantage_sparse, latents, latents_sparse)

        # todo: assign current parameters to old policy; does this work?
        old_param_values = self.policy.get_param_values(trainable=True)
        self.old_policy.set_param_values(old_param_values, trainable=True)

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
