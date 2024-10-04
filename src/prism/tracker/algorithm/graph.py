import itertools
import pathlib
import pickle
from typing import List, Union, Dict

import numpy as np

from .collections import Graph, Step, History


def build_graph(pickle_files: List[Union[str, pathlib.Path]], steps: List[str]) -> Graph:
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
