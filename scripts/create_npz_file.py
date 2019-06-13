import numpy as np
import os
import joblib
from rllab import config
import IPython
from sandbox.finetuning.policies.test_gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from rllab.envs.normalized_env import normalize
import math
from sandbox.finetuning.policies.concurrent_hier_policy import HierarchicalPolicy
# from sandbox.snn4hrl.policies.concurrent_policy_random_time import HierarchicalPolicyRandomTime
import argparse
import re

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_base_dir', '-dir', type=str, default=None)
    parser.add_argument('--combine_dir', '-cdir', action='store_true')
    args = parser.parse_args()


    #manager: manager.pkl
    #skill: skill.pkl
    # want to take manager and skill and combine into single npz file
    if args.combine_dir:
        assert args.pkl_base_dir is not None
        for root, dir, files in os.walk(os.path.join(config.PROJECT_PATH, args.pkl_base_dir)):
            if len(files) > 0:
                # import ipdb;
                # ipdb.set_trace()
                pkl_files = list(filter(lambda x: '.pkl' in x, files))
                sort_nicely(pkl_files)
                assert len(pkl_files) > 1
                assert 'manager.pkl' in pkl_files and 'skill.pkl' in pkl_files
                manager_pkl_path = os.path.join(root, 'manager.pkl')
                manager = joblib.load(manager_pkl_path)['policy']
                manager_params = manager.get_param_values()
                skill_pkl_path = os.path.join(root, 'skill.pkl')
                skill = joblib.load(skill_pkl_path)['policy']
                skill_params = skill.get_param_values()
                # import ipdb; ipdb.set_trace()
                params = {"params": np.concatenate([skill_params, manager_params])}
                npz_path = os.path.join(root, "combined_values.npz")
                np.savez(npz_path, **params)

    else:
        assert args.pkl_base_dir is not None
        for root, dir, files in os.walk(os.path.join(config.PROJECT_PATH, args.pkl_base_dir)):
            if len(files) > 0:
                pkl_files = list(filter(lambda x: '.pkl' in x, files))
                sort_nicely(pkl_files)
                assert len(pkl_files) > 0
                print(pkl_files[-1])
                pkl_path = os.path.join(root, pkl_files[-1])
                policy = joblib.load(pkl_path)['policy']
                if isinstance(policy, HierarchicalPolicy):
                # assert isinstance(policy, GaussianMLPPolicy)
                    params = {"params": policy.get_param_values(hacky_npz=True)}
                else:
                    params = {"params": policy.get_param_values()}
                npz_path = pkl_path[:-4] + "_values.npz"
                np.savez(npz_path, **params)

if __name__ == "__main__":

    main()