"""Generate an NNGP accuracy heatmap over weight_var and bias_var.

For each pair
(weight_var, bias_var), it runs run_experiments.py, reads the test accuracy
written to results.csv, and then saves a heatmap as a PNG.

Example (small 3 x 3 test grid):

  python heat_map_figure.py \
      --weight_vars=1.0,1.5,2.0 \
      --bias_vars=0.0,0.2,0.5 \
      --depth=10 \
      --nonlinearity=relu \
      --num_train=100 \
      --num_eval=1000 \
      --output_file=/nngp/output/phase_diagram.png
"""
from __future__ import print_function

import argparse
import csv
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_float_list(text):
    """Convert '1.0,1.5,2.0' into [1.0, 1.5, 2.0]."""
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def read_test_accuracy(results_file):
    """Read test_acc from the final row produced by run_experiments.py.

    results.csv columns are:
      num_train, nonlinearity, weight_var, bias_var, depth,
      acc_train, acc_valid, acc_test, mse_train, mse_valid, mse_test, final_eps
    """
    with open(results_file, 'r') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise RuntimeError('No results were written to %s' % results_file)
    return float(rows[-1][7])


def run_one(args, weight_var, bias_var, i, j):
    run_dir = os.path.join(args.experiment_dir,
                           'w_%g_b_%g' % (weight_var, bias_var))
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    hparams = ('nonlinearity=%s,depth=%d,weight_var=%g,bias_var=%g' %
               (args.nonlinearity, args.depth, weight_var, bias_var))

    # Use a smaller integration grid to reduce memory use when computing F.
    command = [
        sys.executable,
        args.run_experiments,
        '--num_train=%d' % args.num_train,
        '--num_eval=%d' % args.num_eval,
        '--hparams=%s' % hparams,
        '--experiment_dir=%s' % run_dir,
        '--grid_path=%s' % args.grid_path,
        '--n_gauss=101',
        '--n_var=151',
        '--n_corr=131',
    ]

    print('\n[%d,%d] weight_var=%g, bias_var=%g' %
          (i + 1, j + 1, weight_var, bias_var))
    subprocess.check_call(command)

    results_file = os.path.join(run_dir, 'results.csv')
    accuracy = read_test_accuracy(results_file)
    print('test accuracy = %.4f' % accuracy)
    return accuracy


def make_heatmap(accuracies, weight_vars, bias_vars, output_file,
                 depth, nonlinearity):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(accuracies, origin='lower', aspect='auto')
    fig.colorbar(image, ax=ax, label='Test accuracy')

    ax.set_xticks(np.arange(len(weight_vars)))
    ax.set_xticklabels(['%g' % x for x in weight_vars])
    ax.set_yticks(np.arange(len(bias_vars)))
    ax.set_yticklabels(['%g' % x for x in bias_vars])
    ax.set_xlabel('weight_var')
    ax.set_ylabel('bias_var')
    ax.set_title('NNGP test accuracy: %s, depth=%d' %
                 (nonlinearity, depth))

    for row in range(len(bias_vars)):
        for col in range(len(weight_vars)):
            ax.text(col, row, '%.3f' % accuracies[row, col],
                    ha='center', va='center')

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print('\nSaved heatmap to %s' % output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_vars', default='1.0,1.5,2.0')
    parser.add_argument('--bias_vars', default='0.0,0.2,0.5')
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--nonlinearity',
                    choices=['relu', 'tanh', 'sigmoid'],
                    default='relu')
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--num_eval', type=int, default=1000)
    parser.add_argument('--grid_path', default='./grid_data')
    parser.add_argument('--experiment_dir', default='/tmp/nngp_phase_diagram')
    parser.add_argument('--run_experiments', default='./run_experiments.py')
    parser.add_argument('--output_file',
                        default='/nngp/output/phase_diagram.png')
    args = parser.parse_args()

    weight_vars = parse_float_list(args.weight_vars)
    bias_vars = parse_float_list(args.bias_vars)
    if not weight_vars or not bias_vars:
        raise ValueError('weight_vars and bias_vars must not be empty')

    # Rows = bias_var; columns = weight_var.
    accuracies = np.zeros((len(bias_vars), len(weight_vars)), dtype=float)

    for row, bias_var in enumerate(bias_vars):
        for col, weight_var in enumerate(weight_vars):
            accuracies[row, col] = run_one(
                args, weight_var, bias_var, row, col)

    make_heatmap(accuracies, weight_vars, bias_vars, args.output_file,
                 args.depth, args.nonlinearity)


if __name__ == '__main__':
    main()
