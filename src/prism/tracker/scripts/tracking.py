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
from prism.tracker.algorithm import build_graph, ViterbiTracker


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
    working_dir = task_dir / 'models'
    preprocessed_files = [fp for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    with open(task_dir / 'dataset/steps.txt', 'r') as f:
        steps = [s.strip() for s in f.readlines()]
    graph = build_graph(preprocessed_files, steps)
    print(f'Graph is built for {args.task} using {len(preprocessed_files)} files.')
    print(graph)

    if args.test_pids is None:
        test_pids = [f.stem for f in preprocessed_files]
    else:
        test_pids = args.test_pids.split(',')
    print('Test PIDs:', test_pids)

    with open(working_dir / 'lopo' / 'cm_raw.pkl', 'rb') as f:
        cm_original = pickle.load(f)
    y_test_concatenated, y_pred_concatenated = [], []
    evaluator = Evaluator()

    for test_pid in test_pids:
        print(f'-----Tracking for {test_pid}-----')
        pid_dir = working_dir / 'lopo' / test_pid
        with open(pid_dir / 'pred_raw.pkl', 'rb') as f:
            raw_pred_probas = pickle.load(f)
        with open(pid_dir / 'true.pkl', 'rb') as f:
            trues = pickle.load(f) 
        oracle = None
        if args.n_oracle > 0:
            if args.oracle_times is not None:
                oracle_times = [int(t) for t in args.oracle_times.split(',')]
                assert len(oracle_times) == args.n_oracle, 'Number of oracle times should match n_oracle'
            else:
                oracle_times = None
            oracle = generate_oracle(trues, args.n_oracle, oracle_times)

            print(f'Oracle steps: {oracle}')
            with open(pid_dir / f'oracle_{args.n_oracle}_oracle.pkl', 'wb') as f:
                pickle.dump(oracle, f)

        tracker = ViterbiTracker(graph, confusion_matrix=cm_original, allow_exceptional_transition=False)
        start_time = time.time()
        viterbi_pred_probas = []
        for pred_steps, pred_probas in tracker.predict_batch(raw_pred_probas, oracle=oracle):
            viterbi_pred_probas.append(pred_probas)
        viterbi_pred_probas = np.array(viterbi_pred_probas)
        print(f'Elapsed time: {time.time() - start_time:.2f} sec')
        assert raw_pred_probas.shape == viterbi_pred_probas.shape
        with open(pid_dir / f'pred_viterbi_{args.n_oracle}_oracle.pkl', 'wb') as f:
            pickle.dump(viterbi_pred_probas, f)


        accuracy, f1 = evaluator.frame_level_metrics(trues, np.argmax(viterbi_pred_probas, axis=1))
        with open(pid_dir / f'result_viterbi_{args.n_oracle}_oracle.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1: {f1}\n')

        cm = evaluator.confusion_matrix(trues, np.argmax(viterbi_pred_probas, axis=1), labels=range(len(steps)))
        cm /= cm.sum(axis=1, keepdims=True)
        with open(pid_dir / f'cm_viterbi_{args.n_oracle}_oracle.pkl', 'wb') as f:
            pickle.dump(cm, f)
        plt.imshow(cm, cmap='Blues')
        plt.savefig(pid_dir / f'cm_viterbi_{args.n_oracle}_oracle.png')
        plt.close()
        y_test_concatenated = np.concatenate((y_test_concatenated, trues))
        y_pred_concatenated = np.concatenate((y_pred_concatenated, np.argmax(viterbi_pred_probas, axis=1)))

    if len(y_test_concatenated) > 0:
        accuracy, f1 = evaluator.frame_level_metrics(y_test_concatenated, y_pred_concatenated)
        with open(working_dir / 'lopo' / f'result_viterbi_{args.n_oracle}_oracle.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1: {f1}\n')
        cm = evaluator.confusion_matrix(y_test_concatenated, y_pred_concatenated, labels=range(len(steps)))
        cm /= cm.sum(axis=1, keepdims=True)
        with open(working_dir / 'lopo' / f'cm_viterbi_{args.n_oracle}_oracle.pkl', 'wb') as f:
            pickle.dump(cm, f)
        plt.imshow(cm, cmap='Blues')
        plt.savefig(working_dir / 'lopo' / f'cm_viterbi_{args.n_oracle}_oracle.png')
        plt.close()
    