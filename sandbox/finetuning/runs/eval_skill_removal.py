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





def eval_performance(policy, env, to_remove, max_path_length=5000, num_rollouts=1000):
    policy.manager.to_remove = to_remove
    ext.set_seed(0)
    returns = []
    for i in trange(num_rollouts):
        returns.append(np.sum(rollout(env, policy, max_path_length=max_path_length)['rewards']))
    return returns

if __name__ == "__main__":
    # load the policy
    pkl_paths = {"antgather":"data/local/antlowgeargather-ppo-randominit-trainablelat-latdim6-period10-lr0.003/antlowgeargather_ppo_randominit_trainablelat_latdim6_period10_lr0.003_10/params.pkl",
                 "swimmergather": "data_upload/swimmergather-ppo-randominit-trainablelat-latdim6-period10/swimmergather_ppo_randominit_trainablelat_latdim6_period10_20/params.pkl"}
    env_name = "swimmergather"
    print(env_name)
    pkl_path = pkl_paths[env_name]
    data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
    policy = data['policy']
    env = data['env']


    # collect performance
    returns = []
    for to_remove in range(6):
        returns.append(eval_performance(policy, env, to_remove))
    returns = np.array(returns)
    print(np.mean(returns, axis=1))
    print(np.std(returns, axis=1))
    np.save("{}_skillremoval_returns.npy".format(env_name), returns)
    IPython.embed()



