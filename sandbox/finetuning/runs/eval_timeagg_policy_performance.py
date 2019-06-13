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
from sandbox.finetuning.policies.concurrent_policy_random_time2 import HierarchicalPolicyRandomTime

# mutates the policy, but not in a way that matters
def eval_performance(policy, env, period, max_path_length, num_rollouts, seed=0):
    # import ipdb; ipdb.set_trace()
    # change the policy period
    # do the rollouts and aggregate the performances
    ext.set_seed(seed)
    returns = []
    if isinstance(policy, HierarchicalPolicyRandomTime):
        with policy.fix_period(period):
            for _ in trange(num_rollouts):
                returns.append(np.sum(rollout(env, policy, max_path_length=max_path_length)['rewards']))
        # policy.curr_period = period
        # policy.random_period = False
        # with policy.manager.set_std_to_0():
        # for _ in trange(num_rollouts):
        #     returns.append(np.sum(rollout(env, policy, max_path_length=max_path_length)['rewards']))
    else:
        policy.period = period
        # with policy.manager.set_std_to_0():
        for _ in trange(num_rollouts):
            returns.append(np.sum(rollout(env, policy, max_path_length=max_path_length)['rewards']))
    return returns

# doesn't change anything
def eval_p(policy, env, max_path_length, num_rollouts, seed):
    # change the policy period
    #do the rollouts and aggregate the performances
    ext.set_seed(seed)
    returns = []
    # with policy.manager.set_std_to_0():
    #     for i in trange(num_rollouts):
    #         returns.append(np.sum(rollout(env, policy, max_path_length=max_path_length)['rewards']))
    # return returns
    for i in trange(num_rollouts, desc="Rollouts", ncols=80):
        returns.append(np.sum(rollout(env, policy, max_path_length=max_path_length)['rewards']))
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
    periods = [1, 5, 10, 25, 50, 100]
    # periods = [1]
    returns = []
    for period in tqdm(periods, desc="Period"):
        # print("Period:", period)
        returns.append(eval_performance(policy, env, period, 5000, 100))

    returns = np.array(returns)
    print(np.mean(returns, axis=1))
    print(np.std(returns, axis=1))
    np.save("{}_timestepagg_returns.npy".format(env_name), returns)
    IPython.embed()

def save_latent_info(policy, env, env_name):
    period = 10
    latents = get_latent_info(policy, env, period, 5000, 1000)
    latents = np.concatenate(latents, axis=0)

    # want frequency info
    frequencies = np.sum(latents, axis=0)
    frequencies = frequencies/np.sum(frequencies)

    # want transition info
    transition_matrix = np.zeros([6, 6])
    prev = np.argmax(latents[0])
    for i in range(1, len(latents)):
        next = np.argmax(latents[i])
        transition_matrix[prev, next] += 1
        prev = next
    for i in range(6):
        transition_matrix[i] /= np.sum(transition_matrix[i])

    # save for later
    np.save("{}_latent_frequencies_p10.npy".format(env_name), frequencies)
    np.save("{}_latent_transition_matrix_p10.npy".format(env_name), transition_matrix)
    save_transition_matrix(env_name)
    IPython.embed()

def save_transition_matrix(env_name):
    tm = np.load("{}_latent_transition_matrix_p10.npy".format(env_name))
    classes = ["lat{}".format(str(i)) for i in range(6)]
    make_transition_matrix(tm, classes, title="{} transition matrix".format(env_name))

def make_transition_matrix(tm, classes,
                          normalize=True,
                          title='Transition matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        tm = tm.astype('float') / tm.sum(axis=1)[:, np.newaxis]
        print("Normalized transition matrix")
    else:
        print('Transition matrix, without normalization')

    print(tm)

    plt.imshow(tm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = tm.max() / 2.
    for i, j in itertools.product(range(tm.shape[0]), range(tm.shape[1])):
        plt.text(j, i, format(tm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if tm[i, j] > thresh else "black")

    plt.ylabel('Latent at timestep t')
    plt.xlabel('Latent at timestep t+p')
    plt.tight_layout()
    plt.savefig("{}.png".format(title))

if __name__ == "__main__":
    pkl_paths = {"antgather":"data/local/antlowgeargather-ppo-randominit-trainablelat-latdim6-period10-lr0.003/antlowgeargather_ppo_randominit_trainablelat_latdim6_period10_lr0.003_10/params.pkl",
                 "swimmergather": "data_upload/swimmergather-ppo-randominit-trainablelat-latdim6-period10/swimmergather_ppo_randominit_trainablelat_latdim6_period10_20/params.pkl"}
    env_name = "swimmergather"
    print(env_name)
    pkl_path = pkl_paths[env_name]
    # load the policy
    data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
    policy = data['policy']
    env = data['env']
    # save_return_info(policy, env, "swimmergather")
    # save_latent_info(policy, env, "swimmergather")
    # returns = eval_performance(policy, env, 10, 5000, 1000);
    # save_latent_info(policy, env, env_name)
    save_return_info(policy, env, env_name)
    IPython.embed()
    #rollout(env, policy, max_path_length=max_path_length)



