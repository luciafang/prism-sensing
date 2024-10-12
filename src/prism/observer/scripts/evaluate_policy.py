"""
This script evaluates the observer policy.

before
- datadrive / tasks / {task_name} / observer / lopo / {pid} / dt_distribution.pkl

after
- datadrive / tasks / {task_name} / observer / lopo / {pid} / step_time_dict.pkl
- datadrive / tasks / {task_name} / observer / lopo / {pid} / step_threshold_dict.pkl
"""

import argparse
import numpy as np
import pickle
from tqdm import tqdm

from prism import config
from prism.tracker.algorithm.utils import get_graph
from prism.observer.algorithm import InterventionPolicy, BaselinePolicy


def get_gt_time(ground_truth, target_step):
    for step, gt in ground_truth.items():
        if step.index == target_step.index:
            return gt
    return 0


def find_best_h_threshold(training_data, target_step, threshold_range=np.arange(0.5, 4.0, 0.01)):
    """
    This function finds the best threshold for the h value in the policy.
    
    Args:
    * training_data (List[Dict[str, Dict[Step, List[float]]]): a list of dictionaries containing the expectations and entropys for each step in the procedure.
    * target_step (Step): the target step for the policy.
    * threshold_range (np.ndarray): a range of threshold values to search for the best threshold.

    Returns:
    * best_h_threshold (float): the best threshold for the h value in the policy.
    """
    best_h_threshold = 0
    best_error = 1e9

    for i, h_threshold in enumerate(threshold_range):
        errors = []
        policy = InterventionPolicy(target_step, h_threshold)
        for dt_distribution in training_data:
            expectations = dt_distribution['expectations']
            entropys = dt_distribution['entropys']
            gt_time = get_gt_time(dt_distribution['ground_truth'], target_step)
            triggered_time = policy.predict_batch(expectations, entropys)
            errors.append(abs(triggered_time - gt_time))
        
        if np.mean(np.abs(errors)) < best_error:
            best_error = np.mean(np.abs(errors))
            best_h_threshold = h_threshold
    return best_h_threshold


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Name of the task', required=True)
    parser.add_argument('--test_pids', type=str, help='To specify test pids (e.g., --test_pids 10,11,14)', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    graph = get_graph(args.task)
    if args.test_pids is None:
        test_pids = [fp.stem for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    else:
        test_pids = args.test_pids.split(',')
    print('Test PIDs:', test_pids)

    # grab all pids
    pid_data = {}
    for pid_model_dir in (task_dir / 'models').glob('lopo/*'):
        if not pid_model_dir.is_dir():
            continue
        pid = pid_model_dir.stem
        with open(task_dir / 'observer' / 'lopo' / pid / 'dt_distribution.pkl', 'rb') as f:
            dt_distribution = pickle.load(f)
        pid_data[pid] = dt_distribution
    print('All PIDs:', pid_data.keys())

    for test_pid in test_pids:
        print(f'-----Processing {test_pid}-----')
        training_data = [data for pid, data in pid_data.items() if pid != test_pid]
        test_data = pid_data[test_pid]
        step_time_dict, step_threshold_dict = {}, {}
        for target_step in tqdm(graph.steps[1:-1]):
            # baseline
            baseline_policy = BaselinePolicy(target_step)
            baseline_time = baseline_policy.predict_batch(test_data['expectations'])
            # proposed
            best_h_threshold = find_best_h_threshold(training_data, target_step)
            policy = InterventionPolicy(target_step, best_h_threshold)
            policy_time = policy.predict_batch(test_data['expectations'], test_data['entropys'])
            # gt
            gt_time = get_gt_time(test_data['ground_truth'], target_step)
            step_time_dict[target_step] = {'ground_truth': gt_time, 'proposed': policy_time, 'baseline': baseline_time}
            step_threshold_dict[target_step] = best_h_threshold
        with open(task_dir / 'observer' / 'lopo' / test_pid / 'step_time_dict.pkl', 'wb') as f:
            pickle.dump(step_time_dict, f)
        with open(task_dir / 'observer' / 'lopo' / test_pid / 'step_threshold_dict.pkl', 'wb') as f:
            pickle.dump(step_threshold_dict, f)
        print('results: ', step_time_dict)