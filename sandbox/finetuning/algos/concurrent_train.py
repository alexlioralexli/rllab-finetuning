import theano
import theano.tensor as TT
from sandbox.finetuning.policies.concurrent_hier_policy import HierarchicalPolicy
from sandbox.finetuning.algos.hier_batch_polopt import BatchPolopt  # note that I use my own BatchPolopt class here
from rllab.misc import ext
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import rllab.misc.logger as logger
from sandbox.finetuning.algos.hier_batch_sampler import HierBatchSampler
import numpy as np

class PG_concurrent(BatchPolopt):
    """
    Designed to enable concurrent training of a SNN that parameterizes skills
    and also train the manager at the same time

    Note that, if I'm not trying to do the sample approximation of the weird log of sum term,
    I don't need to know which skill was picked, just need to know the action
    """

    # double check this constructor later
    def __init__(self,
                 manager_optimizer=None,
                 optimizer=None,
                 snn_optimizer=None,
                 optimizer_args=None,
                 step_size=1e-6,
                 latents=None,  # some sort of iterable of the actual latent vectors
                 period=10,  # how often I choose a latent
                 truncate_local_is_ratio=None,
                 **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                # optimizer_args = dict()
                optimizer_args = dict(batch_size=None)
            self.optimizer = FirstOrderOptimizer(learning_rate=step_size, **optimizer_args)  # I hope this is right
        self.manager_optimizer = manager_optimizer
        self.snn_optimizer = snn_optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        super(PG_concurrent, self).__init__(**kwargs) # not sure if this line is correct
        self.latents = latents
        self.period = period

        # todo: fix this sampler stuff
        self.sampler = HierBatchSampler(self, self.period)

        # i hope this is right
        self.diagonal = DiagonalGaussian(self.policy.low_policy.action_space.flat_dim)
        self.debug_fns = []


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

        assert isinstance(self.policy, HierarchicalPolicy)

        # todo: assumptions: 1 trajectory, which is a multiple of p; that the obs_var_probs is valid


        # undoing the reshape, so that batch sampling is ok
        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0]*obs_var_raw.shape[1], obs_var_raw.shape[2]])
        # obs_var = obs_var_raw

        # i, j should contain the probability of latent j at time step self.period*i
        # should be a len(obs)//self.period by len(self.latent) tensor
        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']


        # get the distribution parameters
        # dist_info_vars = []
        # for latent in self.latents:
        #     self.policy.low_policy.set_latent_train(latent)
        #     dist_info_vars.append(self.policy.low_policy.dist_info_sym(obs_var))
        # hopefully the above line takes multiple samples, and state_info_vars not needed as input

        dist_info_vars = self.policy.low_policy.dist_info_sym_all_latents(obs_var)
        probs = [TT.exp(self.diagonal.log_likelihood_sym(action_var, dist_info)) for dist_info in dist_info_vars]

        # need to reshape at the end
        reshaped_probs = [TT.reshape(prob, [obs_var.shape[0]//self.period, self.period]) for prob in probs]

        # now, multiply out each row and concatenate
        subtrajectory_probs = TT.stack([TT.prod(reshaped_prob, axis=1) for reshaped_prob in reshaped_probs], axis=1)
        # shape error might come out of here


        # elementwise multiplication, then sum up each individual row and take log
        likelihood = TT.log(TT.sum(subtrajectory_probs * latent_probs, axis=1))

        surr_loss = - TT.mean(likelihood * advantage_var)

        input_list = [obs_var_raw, obs_var_sparse, action_var, advantage_var]
        # npo has state_info_vars and old_dist_info_vars, I don't think I need them until I go for NPO/TRPO

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            inputs=input_list
        )
        return dict()


    #do the optimization
    def optimize_policy(self, itr, samples_data):
        assert len(samples_data) // self.period == 0

        # note that I have to do extra preprocessing to the advantages, and also create obs_var_sparse

        input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        # print(input_values[0].shape)

        obs_raw = input_values[0].reshape(input_values[0].shape[0]//self.period, self.period, input_values[0].shape[1])
        # obs_raw = input_values[0]

        obs_sparse = input_values[0].take([i for i in range(0, input_values[0].shape[0], self.period)], axis=0)
        advantage_sparse = np.sum(input_values[2].reshape([input_values[2].shape[0]//self.period, self.period]), axis=1)
        all_input_values = (obs_raw, obs_sparse, input_values[1], advantage_sparse)

        loss_before = self.optimizer.loss(all_input_values)
        self.optimizer.optimize(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    def optimize_manager(self, itr, samples_data):
        pass

    def optimize_snn(self, itr, samples_data):
        pass

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env
        )

    def log_diagnostics(self, paths):
        #paths obtained by self.sampler.obtain_samples
        BatchPolopt.log_diagnostics(self, paths)
        # self.sampler.log_diagnostics(paths)   # wasn't doing anything anyways

        # want to log the standard deviations
        # want to log the max and min of the actions

