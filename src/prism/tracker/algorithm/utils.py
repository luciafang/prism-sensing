import itertools
import pathlib
import pickle
from typing import List, Union, Dict

import numpy as np

from .collections import Graph, Step, History
from ... import config


def get_graph(task_name):
    """
    Get Graph object for a task.
    Args:
    * task_name (Str): task name (e.g., latte_making)

    Returns:
    * graph (Graph): a Graph object.
    """
    task_dir = config.datadrive / 'tasks' / task_name
    preprocessed_files = [fp for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    with open(task_dir / 'dataset/steps.txt', 'r') as f:
        steps = [s.strip() for s in f.readlines()]
    graph = _build_graph(preprocessed_files, steps)
    print(f'Graph is built for {task_name=} using {len(preprocessed_files)} files.')
    print(graph)
    return graph


def get_raw_cm(task_name):
    """
    Get raw confusion matrix for a task.
    Args:
    * task_name (Str): task name (e.g., latte_making)

    Returns:
    * cm (List[List[float]]): a list of lists representing the confusion matrix of the task, obtaine through lopo evaluation.
    """
    task_dir = config.datadrive / 'tasks' / task_name
    cm_path = task_dir / 'models' / 'lopo' / 'cm_raw.pkl'
    with open(cm_path, 'rb') as f:
        cm = pickle.load(f)
    print('Confusion matrix is loaded from ', cm_path)
    return cm


def _build_graph(pickle_files: List[Union[str, pathlib.Path]], steps: List[str]) -> Graph:
    """
    This function builds a graph object from a set of pickle files and a list of steps.
    The graph represents transitions between the different steps in a procedure, with the time taken for each step.

    Args:
    * pickle_files (List[Union[str, pathlib.Path]]): a list of the paths to the pickle files containing the input data.
    * steps (List[str]): a list of strings representing the steps in the process.

    Returns:
    * graph (Graph): a graph object with a list of step objects and a dictionary containing transition probabilities.
    """
    steps = ['BEGIN'] + steps + ['END']
    transition_graph = np.zeros((len(steps), len(steps)))
    time_dict: Dict[str, List[int]] = {k: [] for k in steps}
    all_step_index_list = []
    
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as pickle_fp:
            pickle_data = pickle.load(pickle_fp)

        pickle_data['label'] = ['BEGIN'] + pickle_data['label'] + ['END']
        prev_step = None

        step_index_list = []
        for curr_step, group in itertools.groupby(pickle_data['label']):
            if prev_step is not None:
                transition_graph[steps.index(prev_step)][steps.index(curr_step)] += 1

            time_dict[curr_step].append(len(list(group)))
            step_index_list.append(steps.index(curr_step))
            prev_step = curr_step
        all_step_index_list.append(step_index_list)

    step_list = []
    for i, k in enumerate(steps):
        step_list.append(Step(i, k, mean_time=np.nan_to_num(np.mean(time_dict[k])), std_time=np.nan_to_num(np.std(time_dict[k]))))

    edge_dict: Dict[Step, Dict[Step, float]] = {}
    for s in step_list:
        edge_dict[s] = {}
        total = np.sum(transition_graph[s.index])
        for next_step_index in np.nonzero(transition_graph[s.index])[0]:
            edge_dict[s][step_list[next_step_index]] = transition_graph[s.index][next_step_index] / total

    histories = []
    for step_index_list in all_step_index_list:
        history = History([step_list[i] for i in step_index_list])
        histories.append(history)

    return Graph(steps=step_list, edges=edge_dict, histories=histories)
