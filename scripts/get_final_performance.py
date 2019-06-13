import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import sys

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""


def plot_data(data, value="AverageReturn"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    import IPython; IPython.embed()
    sns.set(style="darkgrid", font_scale=1)
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition", linewidth=1, estimator=rolling_mean)
    plt.legend(loc='best').draggable()

def rolling_mean(data, axis=0):
    return pd.rolling_mean(data, 10, axis=1).mean(axis=axis)

def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'progress.csv' in files:
                param_path = open(os.path.join(root, 'params.json'))
                params = json.load(param_path)
                exp_name = params['exp_name'].rsplit('_', 1)[0]

                log_path = os.path.join(root, 'progress.csv')
                try:
                    experiment_data = pd.read_csv(log_path)
                    experiment_data.insert(
                        len(experiment_data.columns),
                        'Unit',
                        unit
                    )
                    experiment_data.insert(
                        len(experiment_data.columns),
                        'Condition',
                        condition or exp_name
                    )

                    datasets.append(experiment_data)
                    unit += 1
                    print(root.split("_")[-1], experiment_data.shape[0])
                except pd.io.common.EmptyDataError:
                    print("Empty Data Error")
    return datasets

def get_value(datasets, value):
    vals = [dataset[value][-25:].mean() for dataset in datasets]
    print(vals)
    return np.mean(vals)

def get_initial_value(datasets, value):
    vals = [dataset[value][0].mean() for dataset in datasets]
    print(vals)
    return np.mean(vals)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str, default=None)
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    parser.add_argument('--get_initial', '-i', action='store_true')
    parser.add_argument('--prev', '-p', type=float, default=None)
    args = parser.parse_args()

    data = get_datasets(args.logdir)
    if args.get_initial:
        assert args.prev is not None
        val = get_initial_value(data, args.value)
        print("First performance: ", val, 100.0*(val-args.prev)/args.prev)
    else:
        print("Average over last 25: ", get_value(data, args.value))


if __name__ == "__main__":
    main()