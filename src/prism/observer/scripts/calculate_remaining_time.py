"""
This script runs the remaining time estimation algorithm.

after
- datadrive / tasks / {task_name} / observer / lopo / {pid} / dt_distribution.pkl
"""

import argparse
import pickle
from tqdm import tqdm

from prism import config
from prism.tracker.algorithm import ViterbiTracker
from prism.tracker.algorithm.utils import get_graph, get_raw_cm
from prism.observer.algorithm import RemainingTimeEstimator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Name of the task', required=True)
    parser.add_argument('--test_pids', type=str, help='To specify test pids (e.g., --test_pids 10,11,14)', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    graph = get_graph(args.task)
    cm = get_raw_cm(args.task)
    if args.test_pids is None:
        test_pids = [fp.stem for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    else:
        test_pids = args.test_pids.split(',')
    print('Test PIDs:', test_pids)

    remaining_time_estimator = RemainingTimeEstimator(graph, mc_samples=1000)
    for test_pid in test_pids:
        print(f'-----Processing {test_pid}-----')
        with open(task_dir / 'models' / 'lopo' / test_pid / 'pred_raw.pkl', 'rb') as f:
            raw_pred_probas = pickle.load(f)
        with open(task_dir / 'models' / 'lopo' / test_pid / 'true.pkl', 'rb') as f:
            y_test = pickle.load(f)  # List[int]
        tracker = ViterbiTracker(graph, confusion_matrix=cm)
        remaining_time_estimator.reset()

        ground_truth = {}  # Dict[Step, int]
        time = 0
        for raw_pred_prob in tqdm(raw_pred_probas):
            _ = tracker.forward(raw_pred_prob)
            _ = remaining_time_estimator.forward(tracker.curr_entries)
            time += 1
            if time == len(y_test):
                break
            if y_test[time - 1] != y_test[time] and graph.steps[y_test[time] + 1] not in ground_truth:
                ground_truth[graph.steps[y_test[time] + 1]] = time

        save_dir = task_dir / 'observer' / 'lopo' / test_pid
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'dt_distribution.pkl', 'wb') as f:
            pickle.dump(
                {'expectations': remaining_time_estimator.expectations,
                'entropys': remaining_time_estimator.entropys,
                'ground_truth': ground_truth},
                f
            )