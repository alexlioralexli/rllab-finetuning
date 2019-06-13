import math
import joblib
from rllab import config
import os
import argparse
import numpy as np
from tqdm import trange, tqdm
import time

# baselines, utilities
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.instrument import run_experiment_lite, stub

# algos
from rllab.algos.vpg import VPG
from rllab.algos.trpo import TRPO
from sandbox.finetuning.algos.hierarchical_vpg import PG_concurrent_approx
from sandbox.finetuning.algos.trpo_snn import TRPO_snn
from sandbox.finetuning.algos.ppo_repeating import PPO_repeating
from sandbox.finetuning.algos.ppo_flat import PPO_flat
from sandbox.finetuning.algos.concurrent_ppo2 import Concurrent_PPO
from sandbox.finetuning.algos.hippo2 import Hippo as HippoRandomTime
from sandbox.finetuning.algos.concurrent_continuous_ppo import ConcurrentContinuousPPO
# policies
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.finetuning.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.finetuning.policies.test_gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.finetuning.policies.concurrent_hier_policy2 import HierarchicalPolicy
from sandbox.finetuning.policies.concurrent_policy_random_time2 import HierarchicalPolicyRandomTime
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
from sandbox.finetuning.envs.mujoco.modified.modified_snake_gather_env import ModifiedSnakeGatherEnv
from sandbox.finetuning.envs.mujoco.modified.snake_hfield_gather import SnakeHfieldGather
from sandbox.finetuning.envs.mujoco.gather.snake_fixedegoobs_gather_env import SnakeFixedEgoObsGatherEnv
from sandbox.finetuning.envs.mujoco.transfer.ant_gather_env_com_penalty import AntLowGearGatherEnvComPenalty
from sandbox.finetuning.envs.mujoco.transfer.ant_gather_env_joint_penalty import AntLowGearGatherEnvJointPenalty
from sandbox.finetuning.envs.mujoco.transfer.snake_gather_env_com_penalty import SnakeGatherEnvComPenalty
from sandbox.finetuning.envs.mujoco.transfer.snake_gather_env_joint_penalty import SnakeGatherEnvJointPenalty
# pchange or eval
from sandbox.finetuning.runs.eval_timeagg_policy_performance import save_return_info, eval_performance, eval_p
from sandbox.finetuning.runs import test_utils
# stub(globals())

'''Launching script for all experiments'''
################################################################
#################### Command Line Arguments ####################
################################################################
parser = argparse.ArgumentParser()
# all
parser.add_argument('--env_name', type=str, default="cartpole")
parser.add_argument('--algo', type=str, default='ppo')
parser.add_argument('--learning_rate', '-lr', type=float, default=None)
parser.add_argument('--batch_size', '-bs', type=int, default=-1)
parser.add_argument('--n_itr', '-n_itr', type=int, default=-1)
parser.add_argument('--discount', '-d', type=float, default=None)
parser.add_argument('--horizon', '-horizon', type=int, default=None)
parser.add_argument('--seed', '-seed', type=int, default=-1)
parser.add_argument('--extra', '-x', type=str, default=None)
parser.add_argument('--mlp_baseline', '-mb', action='store_true')

# exp setup
parser.add_argument('--use_gpu', '-use_gpu', action='store_true')
parser.add_argument('--n_experiments', '-e', type=int, default=-1)
parser.add_argument('--ec2', '-ec2', action='store_true')
parser.add_argument('--test', '-test', action='store_true')  # test, dont overwrite prev experiment

# all ppo variants
parser.add_argument('--train_pi_iters', '-tpi', type=int, default=80)
parser.add_argument('--epsilon', '-eps', type=float, default=0.1)

# hippo
parser.add_argument('--period', '-p', type=int, default=-1)
parser.add_argument('--latdim', '-latdim', type=int, default=-1)
parser.add_argument('--trainable_snn', '-trainsnn', action='store_true')
parser.add_argument('--trainable_vec', '-trainvec', action='store_true')
parser.add_argument('--use_skill_baseline', '-sbl', action='store_true')
parser.add_argument('--mlp_skill_baseline', '-msb', action='store_true')
parser.add_argument('--no_bilinear_integration', '-no_bilinear_integration', action='store_true')
parser.add_argument('--continuous_latent', '-cl', action='store_true')

# hippo random p
parser.add_argument('--min_period', '-minp', type=int, default=1)

# misc for transfer
parser.add_argument('--param_name', '-pn', type=str, default=None)  #name of the param to change
parser.add_argument('--pretrained_manager', '-pretrained_manager', action='store_true')
parser.add_argument('--fixed_manager', '-fixed_manager', action='store_true')
parser.add_argument('--random_init', '-random_init', action='store_true')
parser.add_argument('--pkl_path', '-pkl_path', type=str, default="")
parser.add_argument('--npz_path', '-npz_path', type=str, default="")
parser.add_argument('--manager_path', '-mp', type=str, default=None)
parser.add_argument('--snn_path', '-snnp', type=str, default=None)

parser.add_argument('--pkl_base_dir', '-dir', type=str, default=None)
parser.add_argument('--frozen_layers', '-fl', type=str, default="None")
parser.add_argument('--reinit_layers', '-rl', action='store_true')  # if we freeze layers, reinit weights of not frozen ones

# for testing the period change
parser.add_argument('--pchange', '-pc', action ='store_true')
parser.add_argument('--eval', '-eval', action='store_true')

# for the preference changes
parser.add_argument('--savecomvel', '-scv', action ='store_true')
parser.add_argument('--savejointangles', '-sja', action ='store_true')

# misc
parser.add_argument('--get_exp_paths', '-paths', action='store_true')

########################################################################
# import cProfile
# pr = cProfile.Profile()
# pr.enable()
########################################################################
args = parser.parse_args()
algo_choices = {'vpg', 'hierarchical_vpg', 'hippo', 'hippo_random_p', 'ppo', 'flatppo', 'trpo', 'flattrpo', 'repeatingvpg', 'repeatingppo'}
env_choices = {'cartpole', 'half_cheetah', 'swimmergather', 'swimmergatherhfield', 'antlowgeargather',
               'antnormalgeargather', 'swimmergatherreversed', 'antlowgeargatherreversed', 'swimmer3dgather',
               'swimmer3dgatherhfield', 'snakegather', 'antlowgeargatherhfield', 'crippledantlowgeargather',
               'snakegatherhfield', 'snakefixedegoobsgather', 'antlowgeargatherenvcompenalty', 'antlowgeargatherenvjointpenalty',
               'snakegatherenvcompenalty', 'snakegatherenvjointpenalty'}
algo_name = args.algo
use_gpu = args.use_gpu
assert not use_gpu  # gpu causes problems on local with multiple workers and isn't faster
trainable_snn = args.trainable_snn
skill_baseline = args.use_skill_baseline
train_pi_iters = args.train_pi_iters
epsilon = args.epsilon

bilinear_integration = not args.no_bilinear_integration
if args.ec2:
    mode = 'ec2'
    assert not args.test
else:
    mode = "local"

# random seed parameters
if args.seed == -1:
    seeds = list(range(10, 110, 10))
else:
    seeds = [args.seed]
assert args.n_experiments <= len(seeds)
if args.n_experiments > 0:
    seeds = seeds[:args.n_experiments]


assert algo_name in algo_choices
assert args.env_name in env_choices
assert not (args.random_init and args.pretrained_manager)  # don't do weird and confusing command combos

pkl_path = None
if len(args.pkl_path) > 0:
    assert args.pretrained_manager
    assert not args.random_init

npz_path = None
if len(args.npz_path) > 0:
    assert not args.random_init
    npz_path = args.npz_path

assert args.frozen_layers in {"None", "none", "first", "last"}
if args.frozen_layers == "first":
    freeze_lst = [True, True, False, False]
    reinit_lst = [not freeze for freeze in freeze_lst]
elif args.frozen_layers == "last":
    freeze_lst = [False, False, True, False]
    reinit_lst = [not freeze for freeze in freeze_lst]
else:
    freeze_lst = None
    reinit_lst = None
if args.param_name is not None:
    assert args.param_name in ['None', 'body_mass', 'dof_damping', 'body_inertia', 'geom_friction']

# set up the specific env
if args.env_name == 'cartpole':
    env = normalize(CartpoleEnv())
    snn_pkl_path = None
    manager_pkl_path = None
    n_parallel = 1
    latent_dim = 4
    batch_size = 4000
    max_path_length = 100
    n_itr = 40
    discount = 0.99
    if args.learning_rate is None:
        learning_rate = 0.003
    else:
        learning_rate = args.learning_rate
    if args.period != -1:
        period = args.period
    else:
        period = 4
elif args.env_name in {'swimmergather', 'swimmergatherhfield', 'swimmergatherreversed', 'swimmer3dgather',
                       'snakegather', 'swimmer3dgatherhfield', 'snakegatherhfield', 'snakefixedegoobsgather',
                       'snakegatherenvcompenalty', 'snakegatherenvjointpenalty'}:
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    if args.env_name == 'swimmergather':
        env = normalize(SwimmerGatherEnv(**keyword_args))
        max_path_length = 5e3
    elif args.env_name == 'swimmergatherhfield':
        env = normalize(SwimmerGatherUnevenFloorEnv(**keyword_args))
        max_path_length = 5e3
    elif args.env_name == 'swimmergatherreversed':
        env = normalize(SwimmerGatherReversedEnv(**keyword_args))
        max_path_length = 5e3
    elif args.env_name == 'swimmer3dgatherhfield':
        env = normalize(Swimmer3dHfieldGather(**keyword_args))
        max_path_length = 5e3
    elif args.env_name == 'swimmer3dgather':
        if args.param_name is None:
            env = normalize(Swimmer3dGatherEnv(**keyword_args))
        elif args.param_name == 'body_mass':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((2, 1)), np.ones((2, 1))]).tolist()
            env = normalize(ModifiedSwimmer3dGatherEnv(**keyword_args))
        elif args.param_name == 'dof_damping':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((4, 1)), np.ones((4, 1))]).tolist()
            env = normalize(ModifiedSwimmer3dGatherEnv(**keyword_args))
        elif args.param_name == 'body_inertia':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((2, 3)), np.ones((2, 3))]).tolist()
            env = normalize(ModifiedSwimmer3dGatherEnv(**keyword_args))
        elif args.param_name == 'geom_friction':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([np.ones((4, 3)), 1.5 * np.ones((2, 3)), np.ones((2, 3))]).tolist()
            env = normalize(ModifiedSwimmer3dGatherEnv(**keyword_args))
        else:
            raise NotImplementedError
        max_path_length = 5e3
    elif args.env_name == 'snakegather':
        if args.param_name is None:
            env = normalize(SnakeGatherEnv(**keyword_args))
        elif args.param_name == 'body_mass':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((3, 1)), np.ones((3, 1))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        elif args.param_name == 'dof_damping':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((4, 1)), np.ones((3, 1))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        elif args.param_name == 'body_inertia':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((3, 3)), np.ones((3, 3))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        elif args.param_name == 'geom_friction':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([np.ones((4, 3)), 1.5 * np.ones((3, 3)), np.ones((3, 3))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        elif args.param_name == 'body_mass2':
            keyword_args["param_name"] = 'body_mass'
            keyword_args["multiplier"] = np.concatenate([5 * np.ones((3, 1)), 3* np.ones((3, 1))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        elif args.param_name == 'dof_damping2':
            keyword_args["param_name"] = 'dof_damping'
            keyword_args["multiplier"] = np.concatenate([5 * np.ones((4, 1)), 3*np.ones((3, 1))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        elif args.param_name == 'body_inertia2':
            keyword_args["param_name"] = 'body_inertia'
            keyword_args["multiplier"] = np.concatenate([5 * np.ones((3, 3)), 3*np.ones((3, 3))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        elif args.param_name == 'geom_friction2':
            keyword_args["param_name"] = 'geom_friction'
            keyword_args["multiplier"] = np.concatenate(
                [3*np.ones((4, 3)), 5 * np.ones((3, 3)), 3* np.ones((3, 3))]).tolist()
            env = normalize(ModifiedSnakeGatherEnv(**keyword_args))
        else:
            raise NotImplementedError
        max_path_length = 8e3
    elif args.env_name == 'snakegatherhfield':
        env = normalize(SnakeHfieldGather(**keyword_args))
        max_path_length = 8e3
    elif args.env_name == 'snakefixedegoobsgather':
        env = normalize(SnakeFixedEgoObsGatherEnv(**keyword_args))
        max_path_length = 8e3
    elif args.env_name == 'snakegatherenvcompenalty':
        env = normalize(SnakeGatherEnvComPenalty(**keyword_args))
        max_path_length = 8e3
    elif args.env_name == 'snakegatherenvjointpenalty':
        env = normalize(SnakeGatherEnvJointPenalty(**keyword_args))
        max_path_length = 8e3
    else:
        raise NotImplementedError
    # snn_pkl_path = 'data_upload/egoSwimmer-snn/egoSwimmer-snn_005MI_5grid_6latCat_bil_0030/params.pkl'
    snn_pkl_path = args.snn_path
    assert args.pretrained_manager == (args.manager_path is not None)
    if len(args.pkl_path) > 0:
        pkl_path = args.pkl_path
        snn_pkl_path = None
        manager_pkl_path = None
    elif args.pretrained_manager:
        # manager_pkl_path = "data_upload/hier-snn-egoSwimmer-gather/hier-snn-egoSwimmer-gather6range_10agg_500pl_PREegoSwimmer-snn_005MI_5grid_6latCat_bil_0030_0/params.pkl"
        manager_pkl_path = args.manager_path
    else:
        manager_pkl_path = None
    if args.random_init or args.pkl_base_dir:
        snn_pkl_path = None
        manager_pkl_path = None
    n_parallel = 8 #4
    latent_dim = 6
    batch_size = 5e5

    n_itr = 500
    discount = 0.99
    if args.learning_rate is None:
        learning_rate = 0.003
    else:
        learning_rate = args.learning_rate
    if args.period != -1:
        period = args.period
    else:
        period = 10
elif args.env_name in {'antlowgeargather', 'antnormalgeargather', 'antlowgeargatherreversed', 'antlowgeargatherhfield',
                       'crippledantlowgeargather', 'antlowgeargatherenvcompenalty', 'antlowgeargatherenvjointpenalty'}:
    keyword_args = dict(activity_range=6.0, sensor_range=6.0, sensor_span=math.pi * 2, ego_obs=True)
    if args.env_name == 'antlowgeargather':
        if args.param_name is None:
            env = normalize(AntLowGearGatherEnv(**keyword_args))
        elif args.param_name == 'body_mass' or args.param_name == 'dof_damping':  #share same multiplier shape
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((7, 1)), np.ones((7, 1))]).tolist()
            env = normalize(ModifiedAntLowGearGatherEnv(**keyword_args))
        elif args.param_name == 'body_inertia':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.3 * np.ones((7, 3)), np.ones((7, 3))]).tolist()
            env = normalize(ModifiedAntLowGearGatherEnv(**keyword_args))
        elif args.param_name == 'geom_friction':
            keyword_args["param_name"] = args.param_name
            keyword_args["multiplier"] = np.concatenate([1.5 * np.ones((9, 3)), np.ones((9, 3))]).tolist()
            env = normalize(ModifiedAntLowGearGatherEnv(**keyword_args))
        else:
            raise NotImplementedError
    elif args.env_name == 'antnormalgeargather':
        assert not args.param_name
        env = normalize(AntNormalGearGatherEnv(**keyword_args))
    elif args.env_name == "antlowgeargatherreversed":
        env = normalize(AntLowGearGatherReversedEnv(**keyword_args))
    elif args.env_name == 'antlowgeargatherhfield':
        env = normalize(AntLowGearHfieldGatherEnv(**keyword_args))
    elif args.env_name == 'crippledantlowgeargather':
        env = normalize(CrippledAntLowGearGatherEnv(**keyword_args))
    elif args.env_name == 'antlowgeargatherenvcompenalty':
        env = normalize(AntLowGearGatherEnvComPenalty(**keyword_args))
    elif args.env_name == 'antlowgeargatherenvjointpenalty':
        env = normalize(AntLowGearGatherEnvJointPenalty(**keyword_args))
    else:
        raise NotImplementedError
    snn_pkl_path = args.snn_path
    assert args.pretrained_manager == (args.manager_path is not None)
    if len(args.pkl_path) > 0:
        assert args.pretrained_manager
        assert not args.random_init
        pkl_path = args.pkl_path
        snn_pkl_path = None
        manager_pkl_path = None
    elif args.pretrained_manager:
        manager_pkl_path = args.manager_path
    else:
        manager_pkl_path = None
    if args.random_init or args.pkl_path:
        snn_pkl_path = None
        manager_pkl_path = None
    n_parallel = 8
    latent_dim = 6
    batch_size = 5e5
    max_path_length = 5e3
    n_itr = 500
    discount = 0.99
    if args.learning_rate is None:
        learning_rate = 0.003
    else:
        learning_rate = args.learning_rate
    if args.period != -1:
        period = args.period
    else:
        period = 10
else:
    raise NotImplementedError


if args.mlp_baseline:
    baseline = GaussianMLPBaseline(env_spec=env.spec)
else:
    baseline = LinearFeatureBaseline(env_spec=env.spec)
if args.latdim != -1:
    latent_dim = args.latdim
if args.n_itr != -1:
    n_itr = args.n_itr
if args.discount is not None:
    discount = args.discount
if args.horizon is not None:
    max_path_length = args.horizon
n_itr += 1 # to make it end on a round number

#####################################
######## Experiment Settings ########
#####################################
snapshot_mode = "last"
snapshot_gap = n_itr
algo_lst = []
policy_lst = []
if args.batch_size != -1:
    snapshot_gap = int((n_itr-1) * args.batch_size // batch_size)
    batch_size = args.batch_size
    snapshot_mode = "gap"
    assert (n_itr -1) % snapshot_gap == 0

if args.test:
    batch_size = batch_size//40
    n_parallel = 1
exp_prefix, algo = None, None
if algo_name == 'vpg':
    exp_prefix = args.env_name + "_vpg_larger"
    if args.test:
        exp_prefix = "test_" + exp_prefix
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(256, 64),  # (64, 64)
        min_std=1e-4
    )
    optimizer_args = dict(learning_rate=0.01)
    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,  # 40
        discount=discount,
        optimizer_args=optimizer_args
    )
    exp_prefix = "{0}_lr{1}".format(exp_prefix, str(learning_rate))
elif algo_name == 'hierarchical_vpg':
    exp_common_prefix = args.env_name + "_hiervpg"
    if args.test:
        exp_common_prefix = "test_" + exp_common_prefix
    if skill_baseline:
        exp_common_prefix += "_sbl"
    if args.random_init:
        exp_common_prefix += "_randominit"
    if trainable_snn:
        exp_common_prefix += "_trainablelat"
    else:
        exp_common_prefix += "_fixedlat"
    if args.trainable_vec:
        exp_common_prefix += "_trainablevec"
    else:
        exp_common_prefix += "_fixedvec"
    if args.fixed_manager:
        exp_common_prefix += "_fixedmanager"
    if args.pretrained_manager:
        assert manager_pkl_path is not None
        exp_common_prefix += "_pretrainedmanager"
    if not bilinear_integration:
        exp_common_prefix += "_nobi"

    policy = HierarchicalPolicy(
        env_spec=env.spec,
        env=env,
        pkl_path=pkl_path,
        snn_pkl_path=snn_pkl_path,
        manager_pkl_path=manager_pkl_path,
        latent_dim=latent_dim,
        period=period,
        trainable_snn=trainable_snn,
        trainable_latents=args.trainable_vec,
        trainable_manager=not args.fixed_manager,
        bilinear_integration=bilinear_integration
    )
    algo = PG_concurrent_approx(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=discount,
        step_size=0.01,
        period=period,
        use_skill_dependent_baseline=skill_baseline
    )
    exp_prefix = "{0}_latdim{1}_period{2}_lr{3}".format(exp_common_prefix, str(latent_dim), str(period),
                                                        str(learning_rate))
elif algo_name == "ppo" or algo_name == "hippo" or algo_name == "hippo_random_p":
    exp_common_prefix = args.env_name
    if args.param_name:
        exp_common_prefix += "_" + args.param_name
    exp_common_prefix += "_" + algo_name
    if args.test:
        exp_common_prefix = "test_" + exp_common_prefix
    if args.continuous_latent:
        exp_common_prefix += "_contlat"
    if args.mlp_baseline:
        exp_common_prefix += "_mb"
    else:
        exp_common_prefix += "_lb"
    if skill_baseline and not args.mlp_skill_baseline:
        exp_common_prefix += "_lsbl"
    elif skill_baseline and args.mlp_skill_baseline:
        exp_common_prefix += "_msbl"
    if args.random_init:
        assert snn_pkl_path is None and manager_pkl_path is None and args.pkl_base_dir is None
        # exp_common_prefix += "_randominit"
    elif args.pkl_base_dir is not None:
        exp_common_prefix += "_pretrained"
    elif args.snn_path is not None:
        exp_common_prefix += "_pretrainedskills"
    elif args.pretrained_manager:
        assert (manager_pkl_path is not None) or len(pkl_path) > 0
        exp_common_prefix += "_pretrainedmanager"
    if trainable_snn:
        exp_common_prefix += "_trainablelat"
    else:
        exp_common_prefix += "_fixedlat"
    if args.trainable_vec:
        exp_common_prefix += "_trainablevec"
    else:
        exp_common_prefix += "_fixedvec"
    if args.fixed_manager:
        exp_common_prefix += "_fixedmanager"
    if not bilinear_integration:
        exp_common_prefix += "_nobi"
    if algo_name == "ppo" or algo_name == "hippo":
        policy_params = dict(
            env_spec=env.spec,
            env=env,
            pkl_path=pkl_path,
            snn_pkl_path=snn_pkl_path,
            manager_pkl_path=manager_pkl_path,
            latent_dim=latent_dim,
            period=period,
            trainable_snn=trainable_snn,
            # trainable_latents=args.trainable_vec,
            trainable_manager=not args.fixed_manager,
            bilinear_integration=bilinear_integration,
            continuous_latent=args.continuous_latent
        )
        algo_params = dict(
            env=env,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            discount=discount,
            period=period,
            train_pi_iters=train_pi_iters,
            epsilon=epsilon,
            use_skill_dependent_baseline=skill_baseline,
            mlp_skill_dependent_baseline=args.mlp_skill_baseline,
            freeze_manager= args.fixed_manager,
            freeze_skills = not trainable_snn,
            step_size=learning_rate
        )

        if args.pkl_base_dir is None:
            policy = HierarchicalPolicy(**policy_params)
            if args.continuous_latent:
                algo = ConcurrentContinuousPPO(policy=policy, **algo_params)
            else:
                algo = Concurrent_PPO(policy=policy, **algo_params)
        else:
            for root, dir, files in os.walk(os.path.join(config.PROJECT_PATH, args.pkl_base_dir)):
                if len(files) > 0 and 'pchange_results.npz' not in files:
                    pkl_files = list(filter(lambda x: '.npz' in x, files))
                    assert len(pkl_files) > 0
                    assert 'params_values.npz' in pkl_files or 'itr_2000_values.npz' in pkl_files or 'combined_values.npz' in pkl_files
                    if 'params_values.npz' in pkl_files:
                        f_name = 'params_values.npz'
                    elif 'combined_values.npz' in pkl_files:
                        f_name = 'combined_values.npz'
                    else:
                        f_name = 'itr_2000_values.npz'
                    # policy_params['pkl_path'] = os.path.relpath(os.path.join(root, pkl_files[-1]), config.PROJECT_PATH)
                    policy_params['pkl_path'] = os.path.relpath(os.path.join(root, f_name), config.PROJECT_PATH)
                    print(policy_params['pkl_path'])
                    policy = HierarchicalPolicy(**policy_params)
                    algo = Concurrent_PPO(policy=policy, **algo_params)
                    algo_lst.append((int(root.split("_")[-1]), algo))
                    policy_lst.append(policy)

    elif algo_name == "hippo_random_p":
        policy_params = dict(
            env_spec=env.spec,
            env=env,
            pkl_path=pkl_path,
            snn_pkl_path=snn_pkl_path,
            manager_pkl_path=manager_pkl_path,
            latent_dim=latent_dim,
            min_period=args.min_period,
            max_period=period,
            trainable_snn=trainable_snn,
            # trainable_latents=args.trainable_vec,
            bilinear_integration=bilinear_integration,
            continuous_latent=args.continuous_latent
        )
        algo_params = dict(
            env=env,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            discount=discount,
            average_period=(args.min_period + period) / 2.0,  #todo: changed this 5/13
            train_pi_iters=train_pi_iters,
            epsilon=epsilon,
            use_skill_dependent_baseline=skill_baseline,
            mlp_skill_dependent_baseline=args.mlp_skill_baseline,
            step_size=learning_rate
        )
        if args.pkl_base_dir is None:
            policy = HierarchicalPolicyRandomTime(**policy_params)
            algo = HippoRandomTime(policy=policy, **algo_params)
        else:
            for root, dir, files in os.walk(os.path.join(config.PROJECT_PATH, args.pkl_base_dir)):
                if len(files) > 0 and 'pchange_results.npz' not in files:
                    pkl_files = list(filter(lambda x: '.npz' in x, files))
                    assert len(pkl_files) > 0
                    assert 'params_values.npz' in pkl_files or 'itr_2000_values.npz' in pkl_files
                    if 'params_values.npz' in pkl_files:
                        f_name = 'params_values.npz'
                    else:
                        f_name = 'itr_2000_values.npz'
                    policy_params['pkl_path'] = os.path.relpath(os.path.join(root, f_name), config.PROJECT_PATH)
                    print(policy_params['pkl_path'])
                    policy = HierarchicalPolicyRandomTime(**policy_params)
                    algo = HippoRandomTime(policy=policy, **algo_params)
                    algo_lst.append((int(root.split("_")[-1]), algo))
                    policy_lst.append(policy)

    if algo_name == "ppo" or algo_name == "hippo":
        exp_prefix = "{0}_latdim{1}_period{2}_lr{3}_tpi{4}_eps{5}_disc{6}_bs{7}_h{8}".format(exp_common_prefix,
                                                                                str(latent_dim),
                                                                                str(period),
                                                                                str(learning_rate),
                                                                                str(train_pi_iters),
                                                                                str(epsilon),
                                                                                str(discount),
                                                                                str(int(batch_size)),
                                                                                str(int(max_path_length)))
    elif algo_name == "hippo_random_p":
        exp_prefix = "{0}_latdim{1}_period{2}_{3}_lr{4}_tpi{5}_eps{6}_disc{7}_bs{8}_h{9}".format(exp_common_prefix,
                                                                                    str(latent_dim),
                                                                                    str(args.min_period),
                                                                                    str(period),
                                                                                    str(learning_rate),
                                                                                    str(train_pi_iters),
                                                                                    str(epsilon),
                                                                                    str(discount),
                                                                                    str(int(batch_size)),
                                                                                    str(int(max_path_length)))
elif algo_name == "flatppo":
    exp_common_prefix = args.env_name
    if args.param_name:
        exp_common_prefix += "_" + args.param_name
    exp_common_prefix += "_flatppo"
    if args.test:
        exp_common_prefix = "test_" + exp_common_prefix
    if args.mlp_baseline:
        exp_common_prefix += "_mb"
    else:
        exp_common_prefix += "_lb"
    assert not skill_baseline
    # assert args.random_init
    assert not trainable_snn
    assert not args.trainable_vec
    assert not args.pretrained_manager
    if args.pkl_base_dir is not None or (npz_path is not None and len(npz_path) > 0):
        exp_common_prefix += "_pretrained"
    if freeze_lst is not None:
        exp_common_prefix += "_freeze" + args.frozen_layers
    if reinit_lst is not None:
        exp_common_prefix += "_reinitnotfrozen"
    # assert not bilinear_integration
    policy_params = dict(
        env_spec=env.spec,
        hidden_sizes=(256, 64),  # (64, 64)
        min_std=1e-4,
        npz_path=npz_path,
        freeze_lst=freeze_lst,
        reinit_lst=reinit_lst
    )
    algo_params = dict(
        env=env,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=discount,
        period=period,
        train_pi_iters=train_pi_iters,
        epsilon=epsilon,
        step_size=learning_rate
    )
    if args.pkl_base_dir is None:
        policy = GaussianMLPPolicy(**policy_params)
        algo = PPO_flat(policy=policy, **algo_params)
    else:
        for root, dir, files in os.walk(os.path.join(config.PROJECT_PATH, args.pkl_base_dir)):
            if len(files) > 0 and 'pchange_results.npz' not in files:
                pkl_files = list(filter(lambda x: '.npz' in x, files))
                assert len(pkl_files) > 0
                policy_params['npz_path'] = os.path.relpath(os.path.join(root, pkl_files[-1]), config.PROJECT_PATH)
                print(policy_params['npz_path'])
                policy = GaussianMLPPolicy(**policy_params)
                algo = PPO_flat(policy=policy, **algo_params)
                algo_lst.append((int(root.split("_")[-1]), algo))
                policy_lst.append(policy)
    exp_prefix = "{0}_lr{1}_tpi{2}_eps{3}_disc{4}_bs{5}_h{6}".format(exp_common_prefix,
                                                            str(learning_rate), str(train_pi_iters), str(epsilon),
                                                            str(discount), str(int(batch_size)), str(int(max_path_length)))
elif algo_name == 'trpo':
    exp_common_prefix = args.env_name + "_trpo"
    if args.test:
        exp_common_prefix = "test_" + exp_common_prefix
    if skill_baseline:
        exp_common_prefix += "_sbl"
    assert not args.random_init
    assert not trainable_snn
    assert snn_pkl_path is not None
    exp_common_prefix += "_fixedlat"
    if args.pretrained_manager:
        assert manager_pkl_path is not None
        exp_common_prefix += "_pretrainedmanager"
    if not bilinear_integration:
        exp_common_prefix += "_nobi"
    env = hierarchize_snn(env, time_steps_agg=period, pkl_path=snn_pkl_path)
    policy = CategoricalMLPPolicy(env_spec=env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO_snn(
        env=env,
        policy=policy,
        baseline=baseline,
        self_normalize=True,
        log_deterministic=True,
        batch_size=batch_size / period,
        whole_paths=True,
        max_path_length=max_path_length / period,  # correct for larger envs
        n_itr=n_itr,
        discount=discount,
        step_size=learning_rate,
    )
    exp_prefix = "{0}_latdim{1}_period{2}".format(exp_common_prefix, str(latent_dim), str(period))
elif algo_name == 'flattrpo':
    exp_common_prefix = args.env_name + "_flattrpo"
    if args.test:
        exp_common_prefix = "test_" + exp_common_prefix
    assert not skill_baseline
    # assert args.random_init
    assert not trainable_snn
    assert not args.trainable_vec
    assert not args.pretrained_manager
    assert pkl_path is None
    if npz_path is not None and len(npz_path) > 0:
        exp_common_prefix += "_warmstart"
    if freeze_lst is not None:
        exp_common_prefix += "_freeze" + args.frozen_layers
    if reinit_lst is not None:
        exp_common_prefix += "_reinitnotfrozen"
    # assert not bilinear_integration
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(256, 64),  # (64, 64)
        min_std=1e-4,
        npz_path=npz_path,
        freeze_lst=freeze_lst,
        reinit_lst=reinit_lst
    )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=discount,
        period=period,
        train_pi_iters=train_pi_iters,
        epsilon=epsilon,
        step_size=learning_rate,
        whole_paths=True
    )

    exp_prefix = "{0}_lr{1}_bs{2}".format(exp_common_prefix,str(learning_rate), str(int(batch_size)))
elif algo_name == 'repeatingvpg':
    exp_common_prefix = args.env_name + "_repeatingvpg"
    if args.test:
        exp_common_prefix = "test_" + exp_common_prefix
    assert not args.random_init
    assert not trainable_snn
    assert not args.pretrained_manager
    print("period", period)
    env = ActionRepeatingEnv(env, time_steps_agg=period)

    # env = ActionRepeatingEnv(env, time_steps_agg=period)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(256, 64),  # (64, 64)
        min_std=1e-4
    )
    optimizer_args = dict(learning_rate=0.01)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size / period,
        max_path_length=max_path_length / period,  # correct for larger envs
        n_itr=n_itr,
        discount=discount,
        optimizer_args=optimizer_args
    )
    exp_prefix = "{0}_period{1}".format(exp_common_prefix, str(period))
elif algo_name == 'repeatingppo':
    exp_common_prefix = args.env_name + "_repeatingppo"
    if args.test:
        exp_common_prefix = "test_" + exp_common_prefix
    assert not args.random_init
    assert not trainable_snn
    assert not args.pretrained_manager
    print("period", period)
    env = ActionRepeatingEnv(env, time_steps_agg=period)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # env = ActionRepeatingEnv(env, time_steps_agg=period)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(256, 64),  # (64, 64)
        min_std=1e-4
    )
    algo = PPO_repeating(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size / period,
        max_path_length=max_path_length / period,
        n_itr=n_itr,
        discount=discount,
        period=period,
        train_pi_iters=train_pi_iters,
        epsilon=epsilon,
        step_size=learning_rate
    )
    exp_prefix = "{0}_period{1}_lr{2}".format(exp_common_prefix, str(period), str(learning_rate))

assert exp_prefix is not None
assert algo is not None or len(algo_lst) > 0
if args.extra is not None:
    exp_prefix += "_" + args.extra
if args.get_exp_paths:
    print('data/s3/' + exp_prefix.replace('_', '-'))
else:
    if (not args.eval) and (not args.pchange) and (not args.savecomvel) and (not args.savejointangles):
        if len(algo_lst) > 0:
            # algo_lst = list(filter(lambda x: x[0] != 20 and x[0] != 30, algo_lst))
            # algo_lst = list(filter(lambda x: x[0] != 10, algo_lst))
            print("algo list", algo_lst)
            for seed, algo in algo_lst:
                print(seed)
                exp_name = '{0}_{1}_{2}'.format(exp_prefix, str(seed), time.strftime("%d-%m-%Y_%H-%M-%S"))
                run_experiment_lite(
                    stub_method_call=algo.train(),
                    mode=mode,
                    use_gpu=use_gpu,
                    use_cloudpickle=False,
                    pre_commands=['pip install --upgrade pip'],
                    n_parallel=n_parallel,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                    seed=seed,
                    confirm_remote=False,
                    exp_prefix=exp_prefix,
                    exp_name=exp_name
                )
        else:
            print("seeds", seeds)
            for seed in seeds:
                exp_name = '{0}_{1}_{2}'.format(exp_prefix, str(seed), time.strftime("%d-%m-%Y_%H-%M-%S"))
                run_experiment_lite(
                    stub_method_call=algo.train(),
                    mode=mode,
                    use_gpu=use_gpu,
                    use_cloudpickle=False,
                    pre_commands=['pip install --upgrade pip'],
                    n_parallel=n_parallel,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                    seed=seed,
                    confirm_remote=False,
                    exp_prefix=exp_prefix,
                    exp_name=exp_name
                )
    else:
        if args.eval:
            returns = []
            for i in trange(len(algo_lst)):
                policy = policy_lst[i]
                seed = algo_lst[i][0]
                returns.append(np.mean(eval_p(policy, env, max_path_length, 100, seed)))
            print(returns, np.mean(returns), np.std(returns))
        elif args.pchange:
            periods = [1, 5, 10, 25]
            returns = [list() for _ in range(len(periods))]
            for i, p in enumerate(tqdm(periods, desc='Period', ncols=80, leave=False)):
                for j in trange(len(algo_lst), desc="Seed", ncols=80):
                    policy = policy_lst[j]
                    seed = algo_lst[j][0]
                    returns[i].append(eval_performance(policy, env, p, max_path_length, 100, seed=seed))
            # import ipdb; ipdb.set_trace()
            means = np.array([np.mean(returns[i]) for i in range(len(returns))])
            std_devs = np.array([np.std(returns[i]) for i in range(len(returns))])
            print(means, std_devs)
            results = dict(periods=periods, means=means, std_devs=std_devs)
            np.savez(os.path.join(config.PROJECT_PATH, args.pkl_base_dir, "pchange_results.npz"), results)
        elif args.savecomvel:
            velocities = []
            for j in trange(len(algo_lst), desc="Seed", ncols=80):
                policy = policy_lst[j]
                seed = algo_lst[j][0]
                velocities.extend(test_utils.get_velocities(policy, env, 5000, 5, seed=seed))
        elif args.savejointangles:
            angles = []
            for j in trange(len(algo_lst), desc="Seed", ncols=80):
                policy = policy_lst[j]
                seed = algo_lst[j][0]
                angles.extend(test_utils.get_velocities(policy, env, max_path_length, 5, seed=seed))

        import IPython; IPython.embed()

# algo.train()
# pr.disable()
# pr.dump_stats("profile_hippo2.prof")
