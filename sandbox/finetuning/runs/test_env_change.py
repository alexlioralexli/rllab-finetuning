import numpy as np
import joblib
from rllab.sampler.utils import rollout
import os
from rllab import config
from rllab.misc import ext
from tqdm import trange, tqdm
import IPython
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
from sandbox.finetuning.envs.mujoco.modified.modified_ant_env import ModifiedAntEnv
from sandbox.finetuning.envs.mujoco.modified.modified_ant_gather_env import ModifiedAntLowGearGatherEnv
from rllab.envs.normalized_env import normalize
import math

# mutates the policy, but not in a way that matters
def eval_performance(policy, env, max_path_length, num_rollouts):
    #do the rollouts and aggregate the performances
    ext.set_seed(0)
    returns = []
    with policy.manager.set_std_to_0():
        for i in trange(num_rollouts):
            returns.append(np.sum(rollout(env, policy, max_path_length=max_path_length)['rewards']))
            # if i%50 == 0:
                # print(np.mean(np.array(returns)))
    return returns

def get_latent_info(policy, env, period, max_path_length, num_rollouts):
    # change the policy period
    #do the rollouts and aggregate the performances
    policy.period = period
    ext.set_seed(0)
    latents = []
    for i in trange(num_rollouts):
        latent_infos = rollout(env, policy, max_path_length=max_path_length)['agent_infos']['latents']
        latents.append(latent_infos[np.array(range(0, len(latent_infos), 10), dtype=np.uint32)])
    return latents

def save_return_info(policy, env, env_name):
    periods = [1, 2, 5, 10, 25, 50, 100, 200]
    # periods = [1]
    returns = []
    for period in tqdm(periods, desc="Period"):
        # print("Period:", period)
        returns.append(eval_performance(policy, env, period, 5000, 1000))

    returns = np.array(returns)
    print(np.mean(returns, axis=1))
    print(np.std(returns, axis=1))
    np.save("{}_timestepagg_returns.npy".format(env_name), returns)
    IPython.embed()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('pkl_path', type=str)
    # parser.add_argument('--env_name', type=str, default="cartpole")
    args = parser.parse_args()
    pkl_path = 'data/s3/antlowgeargather-hippo-random-p-randominit-trainablelat-fixedvec-latdim6-period10-lr0.003-tpi10-epsilon0.1-bs100000/antlowgeargather_hippo_random_p_randominit_trainablelat_fixedvec_latdim6_period10_lr0.003_tpi10_epsilon0.1_bs100000_10/itr_1600.pkl'
    data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
    policy = data['policy']
    # env = data['env']
    env = normalize(
        ModifiedAntLowGearGatherEnv(param_name="body_inertia", multiplier=np.concatenate([1.3 * np.ones((7, 3)), np.ones((7, 3))]), activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True))
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    env3 = normalize(ModifiedAntLowGearGatherEnv(**keyword_args))
    keyword_args["param_name"] = "geom_friction"
    keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((9, 3)), np.ones((9, 3))])
    env2 = normalize(ModifiedAntLowGearGatherEnv(**keyword_args))
    # if args.env_name == 'cartpole':
    #     env = normalize(CartpoleEnv())
    #     snn_pkl_path = None
    #     manager_pkl_path = None
    #     n_parallel = 1
    #     latent_dim = 4
    #     batch_size = 4000
    #     max_path_length = 100
    #     n_itr = 40
    #     discount = 0.99
    #     if args.learning_rate < 0:
    #         learning_rate = 0.0003
    #     else:
    #         learning_rate = args.learning_rate
    #     if args.period != -1:
    #         period = args.period
    #     else:
    #         period = 4
    # elif args.env_name in {'swimmergather', 'swimmergatherhfield', 'swimmergatherreversed', 'swimmer3dgather', 'snakegather'}:
    #     if args.env_name == 'swimmergather':
    #         env = normalize(
    #             SwimmerGatherEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True))
    #         max_path_length = 5e3
    #     elif args.env_name == 'swimmergatherhfield':
    #         env = normalize(SwimmerGatherUnevenFloorEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2,
    #                                                     ego_obs=True))
    #         max_path_length = 5e3
    #     elif args.env_name == 'swimmergatherreversed':
    #         env = normalize(
    #             SwimmerGatherReversedEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True))
    #         max_path_length = 5e3
    #     elif args.env_name == 'swimmer3dgather':
    #         env = normalize(
    #             Swimmer3dGatherEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True))
    #         max_path_length = 5e3
    #     elif args.env_name == 'snakegather':
    #         env = normalize(SnakeGatherEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True))
    #         max_path_length = 8e3
    #     else:
    #         raise NotImplementedError
    #     snn_pkl_path = 'data_upload/egoSwimmer-snn/egoSwimmer-snn_005MI_5grid_6latCat_bil_0030/params.pkl'
    #     if len(args.pkl_path) > 0:
    #         pkl_path = args.pkl_path
    #         snn_pkl_path = None
    #         manager_pkl_path = None
    #     elif args.pretrained_manager:
    #         manager_pkl_path = "data_upload/hier-snn-egoSwimmer-gather/hier-snn-egoSwimmer-gather6range_10agg_500pl_PREegoSwimmer-snn_005MI_5grid_6latCat_bil_0030_0/params.pkl"
    #     else:
    #         manager_pkl_path = None
    #     if args.random_init:
    #         snn_pkl_path = None
    #         manager_pkl_path = None
    #     n_parallel = 8  # 4
    #     latent_dim = 6
    #     batch_size = 5e5
    #
    #     n_itr = 500
    #     discount = 0.99
    #     if args.learning_rate < 0:
    #         learning_rate = 0.003
    #     else:
    #         learning_rate = args.learning_rate
    #     if args.period != -1:
    #         period = args.period
    #     else:
    #         period = 10
    # elif args.env_name == 'antlowgeargather' or args.env_name == 'antnormalgeargather' or args.env_name == 'antlowgeargatherreversed':
    #     if args.env_name == 'antlowgeargather':
    #         env = normalize(
    #             AntLowGearGatherEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True))
    #     elif args.env_name == 'antnormalgeargather':
    #         env = normalize(
    #             AntNormalGearGatherEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True))
    #     elif args.env_name == "antlowgeargatherreversed":
    #         env = normalize(
    #             AntLowGearGatherReversedEnv(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2,
    #                                         ego_obs=True))
    #     else:
    #         raise NotImplementedError
    #     snn_pkl_path = None
    #     if len(args.pkl_path) > 0:
    #         assert args.pretrained_manager
    #         assert not args.random_init
    #         pkl_path = args.pkl_path
    #         snn_pkl_path = None
    #         manager_pkl_path = None
    #     elif args.pretrained_manager:
    #         raise NotImplementedError
    #     else:
    #         manager_pkl_path = None
    #     if args.random_init:
    #         snn_pkl_path = None
    #         manager_pkl_path = None
    #     n_parallel = 4
    #     latent_dim = 6
    #     batch_size = 5e5
    #     max_path_length = 5e3
    #     n_itr = 500
    #     discount = 0.99
    #     if args.learning_rate < 0:
    #         learning_rate = 0.003
    #     else:
    #         learning_rate = args.learning_rate
    #     if args.period != -1:
    #         period = args.period
    #     else:
    #         period = 10
    # else:
    #     raise NotImplementedError
    # save_return_info(policy, env, "swimmergather")
    # save_latent_info(policy, env, "swimmergather")
    # returns = eval_performance(policy, env, 10, 5000, 1000);
    # save_latent_info(policy, env, env_name)
    # save_return_info(policy, env, env_name)
    IPython.embed()


if __name__ == "__main__":
    main()



