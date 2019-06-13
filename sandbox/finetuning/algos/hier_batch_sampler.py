from rllab.sampler import parallel_sampler
# from rllab.sampler.base import BaseSampler
from sandbox.finetuning.sampler.hier_base import BaseSampler
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger


class HierBatchSampler(BaseSampler):
    '''
    this class truncates the paths at a multiple of self.period
    '''
    def __init__(self, algo, period=None):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.period = period

    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        raw_paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.period is None:  # hippo random p
            paths = raw_paths
        else:
            #todo: this will break for environments where the rollout terminates after goal is reached
            paths = []
            for path in raw_paths:
                new_length = (len(path['rewards']) // self.period) * self.period
                for key in path.keys():
                    if isinstance(path[key], dict):
                        for key2 in path[key].keys():
                            path[key][key2] = path[key][key2][:new_length]
                    else:
                        path[key] = path[key][:new_length]
                if len(path['rewards']) > 0:
                    paths.append(path)

                # num_padding = self.period - (len(path['rewards']) % self.period)
                # for key in path.keys():
                #     if isinstance(path[key], dict):
                #         for key2 in path[key].keys():
                #             path[key][key2].
            # paths = raw_paths

        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated




    def process_samples(self, itr, paths):
        samples_data = super().process_samples(itr, paths)
        if hasattr(self.algo, "use_skill_dependent_baseline") and self.algo.use_skill_dependent_baseline:
            skill_samples_data = self.process_samples_skill_dependent(itr, paths)
            samples_data['skill_advantages'] = skill_samples_data['advantages']
        return samples_data


    # almost all copied over from hier_base
    def process_samples_skill_dependent(self, itr, paths):
        # need to generate the correct observations using the outer product
        new_paths = []
        for i in range(len(paths)):
            latents = paths[i]['agent_infos']['latents']
            observations = paths[i]['observations']
            # insert the time_remaining
            time_remaining = paths[i]['agent_infos']['time_remaining'].reshape(len(observations), 1)
            extended_obs = np.concatenate([observations, time_remaining], axis=1)
            # new_observations = np.matmul(observations[:, :, np.newaxis], latents[:, np.newaxis, :]).reshape(observations.shape[0], -1)
            new_observations = np.matmul(extended_obs[:, :, np.newaxis], latents[:, np.newaxis, :]).reshape(extended_obs.shape[0], -1)
            new_observations = np.concatenate([new_observations, extended_obs, latents], axis=1)
            new_paths.append(dict(observations=new_observations, rewards=paths[i]['rewards'], returns=paths[i]['returns']))
        paths = new_paths

        baselines = []
        returns = []

        if hasattr(self.algo.skill_dependent_baseline, "predict_n"):
            all_path_baselines = self.algo.skill_dependent_baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.skill_dependent_baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        if not self.algo.policy.recurrent:
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])


            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            samples_data = dict(
                advantages=advantages,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])


            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            samples_data = dict(
                advantages=adv,
            )

        logger.log("fitting skill-depdendent baseline...")
        if hasattr(self.algo.skill_dependent_baseline, 'fit_with_samples'):
            self.algo.skill_dependent_baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.skill_dependent_baseline.fit(paths)
        logger.log("fitted skill-dependent baseline")

        logger.record_tabular('SkillBaselineExplainedVariance', ev)
        return samples_data