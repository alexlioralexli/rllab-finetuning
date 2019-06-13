from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, stub
# from sandbox.snn4hrl.policies.concurrent_hier_policy import HierarchicalPolicy
# from sandbox.snn4hrl.algos.concurrent_ppo import Concurrent_PPO
# from sandbox.snn4hrl.policies.concurrent_hier_policy2 import HierarchicalPolicy
# from sandbox.snn4hrl.algos.concurrent_ppo2 import Concurrent_PPO
# from sandbox.snn4hrl.algos.hippo import Hippo as HippoRandomTime
# from sandbox.snn4hrl.policies.concurrent_policy_random_time import HierarchicalPolicyRandomTime
from sandbox.finetuning.policies.concurrent_policy_random_time2 import HierarchicalPolicyRandomTime
from sandbox.finetuning.algos.hippo2 import Hippo as HippoRandomTime
from sandbox.finetuning.policies.test_gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.finetuning.algos.ppo_flat import PPO_flat

stub(globals())
from tqdm import tqdm
from itertools import product

algo_name = "hippo"

env = normalize(CartpoleEnv())
env_name = "cartpole"
snn_pkl_path = None
manager_pkl_path = None
pkl_path = None
npz_path = None
freeze_lst = None
reinit_lst = None
n_parallel = 1
latent_dim = 4
max_path_length = 100
n_itr = 40
discount = 0.99
period = 4
baseline = LinearFeatureBaseline(env_spec=env.spec)

trainable_snn = True
trainable_vec = False

# batch_sizes = [1000, 2000, 4000]
# lrs = [3e-4, 3e-3, 3e-2]
# epsilons = [0.1, 0.2]
# tpis = [5, 10, 30, 80]
# seeds = [10, 20, 30, 40, 50]
batch_sizes = [5000]
lrs = [3e-4]
epsilons = [0.1]
tpis = [5]
seeds = [40]

for batch_size, lr, epsilon, tpi in list(product(batch_sizes, lrs, epsilons, tpis)):
    if algo_name == "hippo":
        exp_common_prefix = env_name + "_hippo"

        assert snn_pkl_path is None and manager_pkl_path is None
        exp_common_prefix += "_randominit"
        if trainable_snn:
            exp_common_prefix += "_trainablelat"
        else:
            exp_common_prefix += "_fixedlat"
        exp_common_prefix += "_fixedvec"
        # policy = HierarchicalPolicy(
        #     env_spec=env.spec,
        #     env=env,
        #     pkl_path=pkl_path,
        #     snn_pkl_path=snn_pkl_path,
        #     manager_pkl_path=manager_pkl_path,
        #     latent_dim=latent_dim,
        #     period=period,
        #     trainable_snn=trainable_snn
        # )
        # algo = Concurrent_PPO(
        #     env=env,
        #     policy=policy,
        #     baseline=baseline,
        #     batch_size=batch_size,
        #     max_path_length=max_path_length,
        #     n_itr=n_itr,
        #     discount=discount,
        #     period=period,
        #     train_pi_iters=tpi,
        #     epsilon=epsilon,
        #     use_skill_dependent_baseline=False,
        #     step_size=lr
        # )
        # policy = HierarchicalPolicyRandomTime(
        #     env_spec=env.spec,
        #     env=env,
        #     pkl_path=pkl_path,
        #     snn_pkl_path=snn_pkl_path,
        #     manager_pkl_path=manager_pkl_path,
        #     latent_dim=latent_dim,
        #     max_period=period,
        #     trainable_snn=trainable_snn,
        # )
        # algo = HippoRandomTime(
        #     env=env,
        #     policy=policy,
        #     baseline=baseline,
        #     batch_size=batch_size,
        #     max_path_length=max_path_length,
        #     n_itr=n_itr,
        #     discount=discount,
        #     average_period=(1+period)/2.0,
        #     train_pi_iters=tpi,
        #     epsilon=epsilon,
        #     use_skill_dependent_baseline=False,
        #     step_size=lr
        # )
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(256, 64),  # (64, 64)
            min_std=1e-4,
            npz_path=npz_path,
            freeze_lst=freeze_lst,
            reinit_lst=reinit_lst
        )

        algo = PPO_flat(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            discount=discount,
            period=period,
            train_pi_iters=tpi,
            epsilon=epsilon,
            step_size=lr
        )
        exp_prefix = "{0}_latdim{1}_period{2}_lr{3}_tpi{4}_epsilon{5}_bs{6}".format(exp_common_prefix,
                                                                                    str(latent_dim),
                                                                                    str(period),
                                                                                    str(lr),
                                                                                    str(tpi),
                                                                                    str(epsilon), str(int(batch_size)))
        for seed in seeds:
            exp_name = '{0}_{1}'.format(exp_prefix, str(seed))
            run_experiment_lite(
                stub_method_call=algo.train(),
                mode="local",
                use_cloudpickle=False,
                pre_commands=['pip install --upgrade pip'],
                n_parallel=n_parallel,
                snapshot_mode="last",
                seed=seed,
                exp_prefix=exp_prefix,
                exp_name=exp_name,
            )

