import pickle
import numpy as np

from .algorithm import ViterbiTracker, build_graph
from .. import config


class TrackerAPI():

    def __init__(self, task_name, allow_exceptional_transition=False):
        """
        Args:
        * task_name (Str): task name (e.g., latte_making)
        * allow_exceptional_transition (bool): a flag whether to allow exceptional transitions.
        """
        task_dir = config.datadrive / 'tasks' / task_name
        preprocessed_files = [fp for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
        with open(task_dir / 'dataset/steps.txt', 'r') as f:
            steps = [s.strip() for s in f.readlines()]
        graph = build_graph(preprocessed_files, steps)
        print(f'Graph is built for {task_name} using {len(preprocessed_files)} files.')
        print(graph)
        cm_path = task_dir / 'models' / 'lopo' / 'cm_raw.pkl'
        with open(cm_path, 'rb') as f:
            cm = pickle.load(f)
        print('Confusion matrix is loaded from ', cm_path)
        self.tracker = ViterbiTracker(graph, confusion_matrix=cm, allow_exceptional_transition=allow_exceptional_transition)

    def __call__(self, probs):
        """
        This method calculates the Viterbi forward algorithm for a single frame given the current prediction entries.

        Args:
        * probs (List[float]): a list of the observation probabilities of each step at the current frame.

        Returns:
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        self.tracker.forward(probs)
        return self.tracker.__get_probs__(self.tracker.curr_entries.values())
    
    def set_current_step(self, step_index: int):
        """
        This method sets the current step of the tracker.

        Args:
        * step_index (int): an integer representing the step index.
        """
        self.tracker.set_current_step(step_index)

    def get_current_context(self):
        """
        This method returns the current context of the tracker.

        Returns:
        * context (Dict): a dictionary containing the current step index, the probabilities of each step, and the history of steps.
            Keys are 'current_step_index', 'current_step_probs', 'history', and 'potential_next_step_indices'.
        """
        probs = self.tracker.__get_probs__(self.tracker.curr_entries.values())
        probs = list(np.nan_to_num(probs))
        step_index_history = self.tracker.__get_best_history__(self.tracker.curr_entries.values())

        history = []
        prev_step_index = None
        for step_index in step_index_history:
            if prev_step_index is None:
                history.append(step_index)
            elif step_index != prev_step_index:
                history.append(step_index)
            prev_step_index = step_index
        
        ret = {
            'current_step_index': int(np.argmax(probs) + 1),
            'current_step_probs': probs,
            'history': [int(h) for h in history],
            'potential_next_step_indices': [step.index for step in self.tracker.graph.get_potential_next_steps(self.tracker.steps[int(np.argmax(probs) + 1)])]
        }
        return ret
