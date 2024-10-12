import numpy as np
import scipy
from typing import Dict, Iterable, List, Tuple

from prism.tracker.algorithm.collections import Graph, Step


def monte_carlo_estimate(graph: Graph, current_step: Step, current_step_time: int, n_samples: int = 10000) -> Tuple[Dict[Step, float], Dict[Step, List[float]]]:
    """
    This function estimate the expected remaining time for each step based on the given status.

    Args:
    * graph (Graph): a graph object, which represents transitions between the different steps in a procedure.
    * current_step (Step): a step to start the estimation.
    * current_step_time (int): a time spent on the current step before the estimation.
    * n_samples (int): a number of samples to use for the estimation.

    Returns: 
    * expectations (Dict[Step, float]): a dictionary of the expected remaining time for each step.
    * results (Dict[Step, List[float]]): a dictionary of the raw samples for the expected remaining time for each step.
    """
    results = {}
    
    def dfs(current_step: Step, samples: Iterable[float], history: List[Step]):
        accum_prob = 0.0

        for next_step, next_prob in graph.edges[current_step].items():
            if next_step in history:
                continue

            next_samples = samples[int(len(samples) * accum_prob): int(len(samples) * (accum_prob + next_prob))]
            if len(next_samples) == 0:
                continue

            next_samples += scipy.stats.norm.rvs(loc=next_step.mean_time, scale=next_step.std_time, size=len(next_samples))
            results[next_step] = results.get(next_step, []) + list(next_samples)

            dfs(next_step, next_samples, history + [next_step])
            accum_prob += next_prob

    lower = -np.inf if current_step.std_time == 0 else (current_step_time - current_step.mean_time) / current_step.std_time
    samples = scipy.stats.truncnorm.rvs(lower, np.inf, loc=current_step.mean_time, scale=current_step.std_time, size=n_samples) - current_step_time
    dfs(current_step, samples, [current_step])
    
    expectations = {}
    for index, step in enumerate(results):
        expectations[step] = sum(results[step]) / len(results[step])
    
    return expectations, results