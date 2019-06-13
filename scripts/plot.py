import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np

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
COLORS = {"HiPPO, random p": 'green', "HiPPO, p=1": 'red',
          "PPO, Action Repeat p=10": 'blue', "PPO": 'purple',
          "Single Baseline": 'blue', "Manager and Skill Baselines (ours)": 'green',
          'HiPPO from scratch':'blue',
          'HiPPO on trainable skills':'green',
          'HiPPO on fixed skills':'red'
          }


def plot_data(data, value="AverageReturn", remove_penalty=False):
    if isinstance(data, list):
        data = pd.concat([d.rolling(10, min_periods=1, center=True).mean()[:2000] for d in data], ignore_index=True)
    if 'GatherPenalty' in data:
        data['GatherPenalty'] /= data['NumTrajs']
    if remove_penalty:
        data['AverageReturn'] += data['GatherPenalty']
    sns.set(style="darkgrid", font_scale=1.5) # , rc={'figure.figsize':(10, 7.5)})
    # sns.set_palette("Paired")
    sns.tsplot(data=data, time="Iteration", value=value,
               unit="Unit", condition="Condition", linewidth=1.0, color=COLORS)
    plt.ylabel("Average Return")
    # plt.legend(loc='best').remove()
    plt.legend(loc='best').draggable()

def rolling_mean(data, axis=0):
    return pd.rolling_mean(data, 10, min_periods=1, center=True, axis=1).mean(axis=axis)

def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'progress.csv' in files:
            param_path = None
            exp_name = dir
            if 'params.json' in files:
                param_path = open(os.path.join(root, 'params.json'))
            elif 'variant.json' in files:
                param_path = open(os.path.join(root, 'variant.json'))
            if param_path is not None:
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
                print(root.split("_")[-3], root.split("_")[-1], experiment_data.shape[0])
            except pd.io.common.EmptyDataError:
                print("Empty Data Error")
    # get rid of failed ones
    mean_length = np.mean([len(dataset) for dataset in datasets])
    filtered_datasets = [dataset for dataset in datasets if len(dataset) > 0.75*mean_length]
    print(len(filtered_datasets))
    return filtered_datasets

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    parser.add_argument('--baselines', '-bl', default=None, nargs='*')
    parser.add_argument('--title', '-t', type=str, default=None)
    parser.add_argument('--save_path', '-sp', type=str, default=None)
    parser.add_argument('--remove_penalty', '-rp', action='store_true')
    args = parser.parse_args()
    print("--------------------------------------------------------")
    remove_penalty = args.remove_penalty
    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            print(logdir)
            data += get_datasets(logdir, condition=legend_title)
    else:
        for logdir in args.logdir:
            print(logdir)
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]

    sns.set_context("paper")
    sns.set()
    for value in values:
        plot_data(data, value=value, remove_penalty=remove_penalty)
    if args.baselines is not None:  # plot the original performance
        colors = ['blue', 'green', 'red']
        for i in range(len(args.baselines)):
            val, color = args.baselines[i], colors[i]
            plt.axhline(y=float(val), color=color, linestyle='-', linewidth=1)
    if args.title is not None:
        plt.title(args.title)
    if args.save_path is None:
        plt.show()
    else:
        if os.path.isfile(args.save_path):
            if True: # query_yes_no("File at path already exists, overwrite?"):
                plt.savefig(args.save_path)
        else:
            plt.savefig(args.save_path)


if __name__ == "__main__":
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # pr.dump_stats("plot.prof")
    # import IPython; IPython.embed()