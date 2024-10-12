"""
This module contains functions for tracking steps in a procedure using the Viterbi algorithm.

after
- datadrive / tasks / {task_name} / models / lopo / {pid} / pred_viterbi.pkl
- datadrive / tasks / {task_name} / models / lopo / {pid} / cm_viterbi.pkl
- datadrive / tasks / {task_name} / models / lopo / {pid} / cm_viterbi.png
- datadrive / tasks / {task_name} / models / lopo / {pid} / result_viterbi.txt
- datadrive / tasks / {task_name} / models / lopo / cm_viterbi.pkl
- datadrive / tasks / {task_name} / models / lopo / cm_viterbi.png
- datadrive / tasks / {task_name} / models / lopo / result_viterbi.txt
"""

import argparse
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

from prism import config
from prism.har.algorithm import Evaluator
from prism.tracker.algorithm import ViterbiTracker
from prism.tracker.algorithm.utils import get_graph, get_raw_cm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Name of the task', required=True)
    parser.add_argument('--test_pids', type=str, help='To specify test pids (e.g., --test_pids 10,11,14)', default=None)
    parser.add_argument('--n_oracle', type=int, help='Number of oracle steps to generate', default=0)
    parser.add_argument('--oracle_times', type=str, help='Oracle times for each step', default=None)
    return parser.parse_args()


def generate_oracle(y_test, n, oracle_times=None):
    oracle = {}
    indices = np.random.choice(len(y_test), n, replace=False) if oracle_times is None else oracle_times
    for i in indices:
        oracle[i] = y_test[i] + 1  # include BEGIN as step 0
    return oracle


if __name__ == '__main__':
    args = get_args()
    np.random.seed(2024)
    task_dir = config.datadrive / 'tasks' / args.task
    graph = get_graph(args.task)
    raw_cm = get_raw_cm(args.task)
    if args.test_pids is None:
        test_pids = [fp.stem for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    else:
        test_pids = args.test_pids.split(',')
    print('Test PIDs:', test_pids)

    y_test_concatenated, y_pred_concatenated = [], []
    evaluator = Evaluator()

    for test_pid in test_pids:
        print(f'-----Tracking for {test_pid}-----')
        with open(task_dir / 'models' / 'lopo' / test_pid / 'pred_raw.pkl', 'rb') as f:
            raw_pred_probas = pickle.load(f)
        with open(task_dir / 'models' / 'lopo' / test_pid / 'true.pkl', 'rb') as f:
            trues = pickle.load(f) 
        # oracle setup
        oracle = None
        if args.n_oracle > 0:
            if args.oracle_times is not None:
                oracle_times = [int(t) for t in args.oracle_times.split(',')]
                assert len(oracle_times) == args.n_oracle, 'Number of oracle times should match n_oracle'
            else:
                oracle_times = None
            oracle = generate_oracle(trues, args.n_oracle, oracle_times)

            print(f'Oracle steps: {oracle}')
            with open(task_dir / 'models' / 'lopo' / test_pid / f'oracle_{args.n_oracle}_oracle.pkl', 'wb') as f:
                pickle.dump(oracle, f)

        tracker = ViterbiTracker(graph, confusion_matrix=raw_cm, allow_exceptional_transition=False)
        start_time = time.time()
        viterbi_pred_probas = []
        for pred_steps, pred_probas in tracker.predict_batch(raw_pred_probas, oracle=oracle):
            viterbi_pred_probas.append(pred_probas)
        viterbi_pred_probas = np.array(viterbi_pred_probas)
        print(f'Elapsed time: {time.time() - start_time:.2f} sec')
        assert raw_pred_probas.shape == viterbi_pred_probas.shape

        with open(task_dir / 'models' / 'lopo' / test_pid / f'pred_viterbi_{args.n_oracle}_oracle.pkl', 'wb') as f:
            pickle.dump(viterbi_pred_probas, f)
        accuracy, f1 = evaluator.frame_level_metrics(trues, np.argmax(viterbi_pred_probas, axis=1))
        with open(task_dir / 'models' / 'lopo' / test_pid / f'result_viterbi_{args.n_oracle}_oracle.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1: {f1}\n')

        cm = evaluator.confusion_matrix(trues, np.argmax(viterbi_pred_probas, axis=1), labels=range(len(graph.steps) - 2))
        cm /= cm.sum(axis=1, keepdims=True)
        with open(task_dir / 'models' / 'lopo' / test_pid / f'cm_viterbi_{args.n_oracle}_oracle.pkl', 'wb') as f:
            pickle.dump(cm, f)
        plt.imshow(cm, cmap='Blues')
        plt.savefig(task_dir / 'models' / 'lopo' / test_pid / f'cm_viterbi_{args.n_oracle}_oracle.png')
        plt.close()
        y_test_concatenated = np.concatenate((y_test_concatenated, trues))
        y_pred_concatenated = np.concatenate((y_pred_concatenated, np.argmax(viterbi_pred_probas, axis=1)))

    # aggregate results
    if len(y_test_concatenated) > 0:
        accuracy, f1 = evaluator.frame_level_metrics(y_test_concatenated, y_pred_concatenated)
        with open(task_dir / 'models' / 'lopo' / f'result_viterbi_{args.n_oracle}_oracle.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1: {f1}\n')
        cm = evaluator.confusion_matrix(y_test_concatenated, y_pred_concatenated, labels=range(len(graph.steps) - 2))
        cm /= cm.sum(axis=1, keepdims=True)
        with open(task_dir / 'models' / 'lopo' / f'cm_viterbi_{args.n_oracle}_oracle.pkl', 'wb') as f:
            pickle.dump(cm, f)
        plt.imshow(cm, cmap='Blues')
        plt.savefig(task_dir / 'models'/ 'lopo' / f'cm_viterbi_{args.n_oracle}_oracle.png')
        plt.close()
    