import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from sandbox.finetuning.sampler.utils_snn import rollout_snn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=400,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--switch_every', type=float, default=200,
                        help='Switch latent every')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        while True:
            path = rollout_snn(env, policy, max_path_length=args.max_path_length,
                               switch_lat_every=args.switch_every,
                               animated=True, speedup=args.speedup)
