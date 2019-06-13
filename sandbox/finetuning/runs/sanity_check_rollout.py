# from sandbox.snn4hrl.policies.hier_snn_mlp_policy import GaussianMLPPolicy_snn_hier
from sandbox.finetuning.policies.test_hier_snn_mlp_policy import GaussianMLPPolicy_snn_hier
from rllab.envs.normalized_env import normalize
from sandbox.finetuning.envs.mujoco.swimmer_env import SwimmerEnv
from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from sandbox.finetuning.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.finetuning.envs.hierarchized_snn_env import hierarchize_snn
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.finetuning.policies.concurrent_hier_policy import HierarchicalPolicy
from sandbox.finetuning.algos.concurrent_train import PG_concurrent
# from sandbox.snn4hrl.algos.approx_concurrent_train import PG_concurrent_approx
from sandbox.finetuning.algos.hierarchical_vpg import PG_concurrent_approx
from rllab import config
import numpy as np
from rllab.misc.instrument import stub, run_experiment_lite
import os
import time
from sandbox.finetuning.sampler.utils import rollout
from rllab.envs.box2d.cartpole_env import CartpoleEnv
import joblib
import math
from sandbox.finetuning.envs.action_repeating_env import ActionRepeatingEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.algos.vpg import VPG
from sandbox.finetuning.envs.mujoco.gather.ant_low_gear_gather_env import AntLowGearGatherEnv

# snn_pkl_path = 'data/s3/egoSwimmer-snn/egoSwimmer-snn_005MI_5grid_6latCat_bil_0030/params.pkl'
# manager_pkl_path = 'data_upload/hier-snn-egoSwimmer-gather/hier-snn-egoSwimmer-gather6range_10agg_500pl_PREegoSwimmer-snn_005MI_5grid_6latCat_bil_0030_0/params.pkl' # "sandbox/snn4hrl/runs/manager_params.pkl"
#
# env = normalize(SwimmerGatherEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi*2, ego_obs=True))
# env = normalize(CartpoleEnv())
#
# # Step 1: load the SNN correctly
# policy = HierarchicalPolicy(
#     env_spec=inner_env.spec,
#     env=inner_env,
#     # snn_pkl_path=snn_pkl_path, #snn_pkl_path,
#     # manager_pkl_path=manager_pkl_path, # manager_pkl_path,
#     latent_dim=6, # 6
#     period=10,
#     trainable_snn=True,
#     trainable_latents=True
# )

# rewards = []
# paths = []
# for i in range(1):
#     print("iteration ", i)
#     path_dict = rollout(inner_env, policy, 5000, speedup=1000)
#     rewards.append(sum(path_dict['rewards']))
#     paths.append(path_dict)
# print(rewards)
# env = normalize(AntLowGearGatherEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi*2, ego_obs=True))
# policy = HierarchicalPolicy(
#     env_spec=env.spec,
#     env=env,
#     pkl_path="data/local/antlowgeargather-ppo-randominit-trainablelat-latdim6-period10-lr0.003/antlowgeargather_ppo_randominit_trainablelat_latdim6_period10_lr0.003_10/params.pkl",
#     latent_dim=6,
#     period=10,
#     trainable_snn=True,
#     trainable_latents=True,
# )
# pkl_path = "data/s3/swimmergather-flatppo-lr0.003-tpi10-epsilon0.1-bs500000.0/swimmergather_flatppo_lr0.003_tpi10_epsilon0.1_bs500000.0_10/params.pkl"
# pkl_path= "data/local/test-cartpole-ppo-fixedlat-fixedvec-latdim4-period4-lr0.0003-tpi80-epsilon0.1-bs4000/test_cartpole_ppo_fixedlat_fixedvec_latdim4_period4_lr0.0003_tpi80_epsilon0.1_bs4000_10/params.pkl"
# pkl_path = "data/local/antlowgeargather-ppo-randominit-trainablelat-latdim6-period10-lr0.003/antlowgeargather_ppo_randominit_trainablelat_latdim6_period10_lr0.003_10/params.pkl"
pkl_path = "data_upload/snn-egoNewAnt-400pl_200Switch_2H_0Visit_5Surv_005dist_100GridB_10After_6latent_Bil_50000bs_400pl_0005_lowgear/params.pkl"
data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
policy = data['policy']
env = data['env']
import IPython; IPython.embed()

##########################################################
# while True:
#     obs = env.reset()
#     for _ in range(1000):
#         env.render()
#         action = policy.get_action(obs)[0]
#         print(action)
#         obs, reward, done, info= env.step(action)
###################################################
# import IPython; IPython.embed()
# env = normalize(CartpoleEnv())
# max_path_length = 100
# n_itr = 40
# discount = 0.99
# period = 1
# print("period", period)
# env = ActionRepeatingEnv(env, time_steps_agg=period)
#
# # env = ActionRepeatingEnv(env, time_steps_agg=period)
# policy = GaussianMLPPolicy(
#     env_spec=env.spec,
#     hidden_sizes=(256, 64),  # (64, 64)
#     min_std=1e-4
# )
# optimizer_args = dict(learning_rate=0.01)
# baseline = LinearFeatureBaseline(env_spec=env.spec)
#
# rewards = []
# paths = []
# for i in range(1):
#     print("iteration ", i)
#     path_dict = rollout(env, policy, 100, speedup=1000)
#     rewards.append(sum(path_dict['rewards']))
#     paths.append(path_dict)
# print(rewards)
# import IPython; IPython.embed()