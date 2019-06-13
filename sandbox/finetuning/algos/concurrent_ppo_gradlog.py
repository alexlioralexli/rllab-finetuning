import theano
import theano.tensor as TT
from rllab.misc import ext
import numpy as np
import rllab.misc.logger as logger
from sandbox.finetuning.algos.concurrent_ppo2 import Concurrent_PPO


class Concurrent_PPO_Gradient_Info(Concurrent_PPO):

    def init_opt(self):
        self.init_grad_approx_infos()
        super(Concurrent_PPO_Gradient_Info, self).init_opt()

    def init_grad_approx_infos(self):
        # variables
        obs_var_raw = ext.new_tensor('obs', ndim=3, dtype=theano.config.floatX)
        obs_var_sparse = ext.new_tensor('sparse_obs', ndim=2, dtype=theano.config.floatX)
        action_var = self.env.action_space.new_tensor_variable('action', extra_dims=1,)  # this is 5k?
        # this will have to be the advantage every self.period timesteps
        advantage_var = ext.new_tensor('advantage', ndim=1, dtype=theano.config.floatX)
        advantage_var_sparse = ext.new_tensor('sparse_advantage', ndim=1, dtype=theano.config.floatX)  # this is 5000
        latent_var_sparse = ext.new_tensor('sparse_latent', ndim=2, dtype=theano.config.floatX)
        latent_var = ext.new_tensor('latents', ndim=2, dtype=theano.config.floatX)   # this is 5000
        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])
        matrix = TT.eye(self.num_latents)
        latent_vectors = [matrix[i:i+1, :] for i in range(self.num_latents)]


        # should be a len(obs)//self.period by len(self.latent) tensor
        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
        dist_info_vars = [self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent.repeat(obs_var.shape[0], axis=0)) for latent in latent_vectors]
        logprobs = [self.diagonal.log_likelihood_sym(action_var, dist_info) for dist_info in dist_info_vars]

        # need to reshape at the end
        reshaped_logprobs = [TT.reshape(prob, [obs_var.shape[0]//self.period, self.period]) for prob in logprobs]
        # now, multiply out each row and concatenate
        subtrajectory_logprobs = TT.stack([TT.sum(reshaped_prob, axis=1) for reshaped_prob in reshaped_logprobs], axis=1)

        # exact loss
        subtrajectory_probs = TT.exp(subtrajectory_logprobs)
        likelihood = TT.log(TT.sum(subtrajectory_probs * latent_probs, axis=1))
        surr_loss_exact = - TT.mean(likelihood * advantage_var_sparse   )

        # approximate
        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)
        manager_surr_loss = - TT.mean(TT.log(actual_latent_probs) * advantage_var_sparse)
        dist_info_approx = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
        actual_action_log_probs = self.diagonal.log_likelihood_sym(action_var, dist_info_approx)
        skill_surr_loss = - TT.mean(actual_action_log_probs * advantage_var)
        surr_loss_approx = manager_surr_loss / self.period + skill_surr_loss

        input_list = [obs_var_raw, obs_var_sparse, action_var, advantage_var, advantage_var_sparse, latent_var, latent_var_sparse]
        grad_exact = theano.grad(surr_loss_exact, self.policy.get_params(trainable=True), disconnected_inputs='ignore')
        grad_approx = theano.grad(surr_loss_approx, self.policy.get_params(trainable=True), disconnected_inputs='ignore')
        grad_exact = [grad.flatten() for grad in grad_exact]
        grad_approx = [grad.flatten() for grad in grad_approx]
        v1 = TT.concatenate(grad_exact, axis=0) + 1e-8
        v2 = TT.concatenate(grad_approx, axis=0) + 1e-8
        v1 = v1 / TT.sqrt(TT.sum(TT.sqr(v1)))
        v2 = v2 / TT.sqrt(TT.sum(TT.sqr(v2)))

        cosine_distance = TT.sum(v1 * v2)
        actual_subtrajectory_prob = TT.sum(subtrajectory_probs * latent_var_sparse, axis=1)
        proportion = TT.mean(actual_subtrajectory_prob / TT.sum(subtrajectory_probs, axis=1))

        self.get_dist_infos = ext.compile_function(inputs=input_list, outputs=dist_info_vars[0]['mean'])
        self.get_logprobs = ext.compile_function(inputs=input_list, outputs=logprobs[0])
        self.get_subprobs = ext.compile_function(inputs=input_list, outputs=[subtrajectory_probs, actual_subtrajectory_prob])
        self.get_likelihood = ext.compile_function(inputs=input_list, outputs=[likelihood])
        self.get_surr_loss_exact = ext.compile_function(inputs=input_list, outputs=[surr_loss_exact])
        self.get_surr_loss_approx = ext.compile_function(inputs=input_list, outputs=[surr_loss_approx])
        self.get_vs = ext.compile_function(inputs=input_list, outputs=[v1, v2])
        self.get_gradient_infos = ext.compile_function(inputs=input_list, outputs=[cosine_distance, proportion])
        return dict()

    def optimize_policy(self, itr, samples_data):
        print(len(samples_data['observations']), self.period)
        assert len(samples_data['observations']) % self.period == 0

        # note that I have to do extra preprocessing to the advantages, and also create obs_var_sparse

        if self.use_skill_dependent_baseline:
            input_values = tuple(ext.extract(
                samples_data, "observations", "actions", "advantages", "agent_infos", "skill_advantages"))
        else:
            input_values = tuple(ext.extract(
                samples_data, "observations", "actions", "advantages", "agent_infos"))

        obs_raw = input_values[0].reshape(input_values[0].shape[0] // self.period, self.period,
                                          input_values[0].shape[1])

        obs_sparse = input_values[0].take([i for i in range(0, input_values[0].shape[0], self.period)], axis=0)
        advantage_sparse = input_values[2].reshape([input_values[2].shape[0] // self.period, self.period])[:, 0]
        latents = input_values[3]['latents']
        latents_sparse = latents.take([i for i in range(0, latents.shape[0], self.period)], axis=0)
        mean = input_values[3]['mean']
        log_std = input_values[3]['log_std']
        prob = np.array(
            list(input_values[3]['prob'].take([i for i in range(0, latents.shape[0], self.period)], axis=0)),
            dtype=np.float32)
        if self.use_skill_dependent_baseline:
            advantage_var = input_values[4]
        else:
            advantage_var = input_values[2]
        if self.freeze_skills and not self.freeze_manager:
            all_input_values = (obs_sparse, advantage_sparse, latents_sparse, prob)
        elif self.freeze_manager and not self.freeze_skills:
            all_input_values = (obs_raw, input_values[1], advantage_var, latents, mean, log_std)
        else:
            assert (not self.freeze_manager) or (not self.freeze_skills)
            all_input_values = (obs_raw, obs_sparse, input_values[1], advantage_var, advantage_sparse, latents,
                            latents_sparse, mean, log_std, prob)

        loss_before = self.optimizer.loss(all_input_values)
        self.optimizer.optimize(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        cosine_distance, proportion = 0.0, 0.0
        if self.current_itr % 1 == 0:
            cosine_distance, proportion = self.get_gradient_infos(obs_raw, obs_sparse, input_values[1], advantage_var, advantage_sparse, latents,
                            latents_sparse)
        logger.record_tabular("Cosine Distance", cosine_distance)
        logger.record_tabular("Proportion", proportion)
        return dict()