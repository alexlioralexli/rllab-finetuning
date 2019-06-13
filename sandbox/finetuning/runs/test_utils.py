
# envs
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from sandbox.finetuning.envs.mujoco.gather.swimmer3d_gather_env import Swimmer3dGatherEnv
from sandbox.finetuning.envs.mujoco.gather.snake_gather_env import SnakeGatherEnv
from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_reversed_reward_env import SwimmerGatherReversedEnv
from sandbox.finetuning.envs.mujoco.gather.swimmer_gather_unevenfloor_env import SwimmerGatherUnevenFloorEnv
from sandbox.finetuning.envs.mujoco.gather.ant_low_gear_gather_env import AntLowGearGatherEnv
from sandbox.finetuning.envs.mujoco.gather.ant_low_gear_gather_reversed_reward_env import AntLowGearGatherReversedEnv
from sandbox.finetuning.envs.mujoco.gather.ant_normal_gear_gather_env import AntNormalGearGatherEnv
from sandbox.finetuning.envs.hierarchized_snn_env import hierarchize_snn
# from sandbox.snn4hrl.envs.action_repeating_env import ActionRepeatingEnv
from sandbox.finetuning.envs.action_repeating_env2 import ActionRepeatingEnv
from sandbox.finetuning.envs.mujoco.modified.modified_ant_gather_env import ModifiedAntLowGearGatherEnv
from sandbox.finetuning.envs.mujoco.modified.modified_swimmer3d_gather_env import ModifiedSwimmer3dGatherEnv
from sandbox.finetuning.envs.mujoco.modified.swimmer3d_hfield_gather import Swimmer3dHfieldGather
from sandbox.finetuning.envs.mujoco.modified.low_gear_ant_hfield_gather_env import AntLowGearHfieldGatherEnv
from sandbox.finetuning.envs.mujoco.modified.crippled_low_gear_ant_gather_env import CrippledAntLowGearGatherEnv
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



def get_flatppo_hidden_sizes(env):
    raise NotImplementedError


# returns a list with length num_rollouts; each element is a numpy array of length max_path_length with the velocity
def get_velocities(policy, env, max_path_length, num_rollouts, seed=0):
    ext.set_seed(seed)
    velocities = []
    for _ in trange(num_rollouts):
        rollout_result = rollout(env, policy, max_path_length=max_path_length)
        velocities.append(rollout_result['env_infos']['com_velocity_value'])
    return velocities

def get_velocities(policy, env, max_path_length, num_rollouts, seed=0):
    ext.set_seed(seed)
    angles = []
    for _ in trange(num_rollouts):
        rollout_result = rollout(env, policy, max_path_length=max_path_length)
        angles.append(rollout_result['env_infos']['joint_angles'])
    return angles