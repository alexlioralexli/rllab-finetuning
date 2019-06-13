import theano
import theano.tensor as TT
from sandbox.finetuning.policies.concurrent_hier_policy import HierarchicalPolicy
from sandbox.finetuning.algos.hier_batch_polopt import BatchPolopt  # note that I use my own BatchPolopt class here
from rllab.misc import ext
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import rllab.misc.logger as logger
from sandbox.finetuning.algos.hier_batch_polopt import BatchSampler
import copy
from rllab.spaces.box import Box
from rllab.envs.env_spec import EnvSpec
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline


class PPO_flat(BatchPolopt):
    """
    Normal clipped PPO version of the HiPPO one that this paper investigates.
    """

    # double check this constructor later
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.0003,
                 truncate_local_is_ratio=None,
                 epsilon=0.1,
                 train_pi_iters=80,
                 **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(batch_size=None)
            self.optimizer = FirstOrderOptimizer(learning_rate=step_size, max_epochs=train_pi_iters,
                                                 **optimizer_args)
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.epsilon = epsilon

        super(PPO_flat, self).__init__(**kwargs)  # not sure if this line is correct

        # i hope this is right
        self.debug_fns = []
        # self.old_policy = copy.deepcopy(self.policy)

    # initialize the computation graph
    # optimize is run on >= 1 trajectory at a time

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1,
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

        old_dist_info_vars = dict(mean=mean_var, log_std=log_std_var)
        dist_info_vars = self.policy.dist_info_sym(obs_var)
        lr = self.policy.distribution.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        surr_loss_vector = TT.minimum(lr * advantage_var,
                                      TT.clip(lr, 1 - self.epsilon, 1 + self.epsilon) * advantage_var)
        surr_loss = -TT.mean(surr_loss_vector)

        input_list = [obs_var, action_var, advantage_var, mean_var, log_std_var]

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            inputs=input_list
        )
        return dict()

    # do the optimization
    def optimize_policy(self, itr, samples_data):
        # note that I have to do extra preprocessing to the advantages, and also create obs_var_sparse

        input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages", "agent_infos"
        ))
        mean = input_values[3]['mean']
        log_std = input_values[3]['log_std']

        all_input_values = (input_values[0], input_values[1], input_values[2], mean, log_std)
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
