import copy
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .collections import Graph, HiddenState, HiddenTransition, ViterbiEntry, History


class ViterbiTracker:


    def __init__(self,
                 graph: Graph,
                 confusion_matrix: List[List[float]],
                 allow_exceptional_transition: bool = False,
                 min_frames_per_step: int = 100):
        """
        Args:
        * graph (Graph): a graph object built using build_graph(), which represents transitions between the different steps in a procedure.
        * allow_exceptional_transition (bool): a flag whether to allow exceptional transitions.
        * min_frames_per_step (int): the tracker considers at least 100 frames for each step.
        """
        self.graph = graph
        self.steps = sorted(graph.steps, key=lambda step: step.index)
        self.original_cm = confusion_matrix
        self.allow_exceptional_transition = allow_exceptional_transition
        self.min_frames_per_step = min_frames_per_step

        # initialize: we know the first step is s0: BEGIN
        self.curr_entries: Dict[int, ViterbiEntry] = {}  # step index -> entry
        self.curr_entries[0] = ViterbiEntry(0.0, history=[HiddenState(0, 0)])
        self.curr_time = 0

        print('Tracker initialized.')


    def _get_context_aware_transitions(self, curr_entry: ViterbiEntry) -> List[HiddenTransition]:
        """
        This method returns the possible transitions from the current step based on the history context.
        Args:
        * curr_entry (ViterbiEntry): a ViterbiEntry object.

        Returns:
        * transitions (List[HiddenTransition]): a list of HiddenTransition objects.
        """
        
        # history
        curr_history = []
        prev_step = None
        for state in curr_entry.history:
            step = self.steps[state.step_index]
            if prev_step is None:
                curr_history.append(step)
            elif step != prev_step:
                curr_history.append(step)
            prev_step = step
        curr_history = History(curr_history)

        # 3 sigma can cover almost 100% of the transition
        curr_step = self.steps[curr_entry.last_state.step_index]
        max_time = max(self.min_frames_per_step, int(curr_step.mean_time + curr_step.std_time * 3))
        # cdf represents the (reversed) probability of staying on the step at the time
        probs = 1 - stats.norm.cdf(range(max_time), loc=curr_step.mean_time, scale=curr_step.std_time)
        transitions: List[HiddenTransition] = []

        # get potential histories
        potential_histories = self.graph.get_potential_histories(curr_history)

        if len(potential_histories) == 0:
            if not self.allow_exceptional_transition:
                print('[warning] No potential histories found and no exceptional transition is allowed. Please check the input data.')
                print('curr_history:', curr_history)
                return []
            else:  # follow the most frequent transition from the current step
                edges_from_curr_step = self.graph.edges.get(curr_step, {})
        else:
            edges_from_curr_step = {}
            for step in self.steps:
                candidate = curr_history.steps + [step]
                transition_prob = 0
                for potential_history, p in potential_histories.items():
                    if len(candidate) > len(potential_history.steps):
                        continue
                    if all([candidate[i] == potential_history.steps[i] for i in range(len(candidate))]):
                        transition_prob += p / sum(potential_histories.values())
                if transition_prob != 0:
                    edges_from_curr_step[step] = transition_prob

        time = curr_entry.last_state.time
        if time >= len(probs) - 1 or curr_step.index == len(self.steps) - 1:
            escape_prob = sys.float_info.epsilon
        elif curr_step.index == 0:
            escape_prob = 1.0 - sys.float_info.epsilon
        else:
            escape_prob = 1.0 - probs[time + 1] / probs[time]

        transitions.append(HiddenTransition(curr_step.index, np.log(1 - escape_prob)))
        for dest_step, dest_prob in edges_from_curr_step.items():  # transition to other steps
            transitions.append(HiddenTransition(dest_step.index, np.log(escape_prob * dest_prob)))

        return transitions
    

    def _get_context_aware_confusion_matrix(self) -> List[float]:
        # implement this method, using cetainly-visited steps  --> not needed as it is already considered in the transition
        return self.original_cm


    def __get_best_history__(self, entries: Iterable[ViterbiEntry]) -> Tuple[float, List[int]]:
        """
        This method selects the best entry from a list of ViterbiEntry based on probability.

        Args:
        * entries (Iterable[ViterbiEntry]): a list of ViterbiEntry to choose from.

        Returns:
        * steps (List[int]): a list of integers representing the step indices in the best entry's history.
        """
        entries = sorted(entries, key=lambda entry: entry.log_prob, reverse=True)
        return [state.step_index for state in entries[0].history]
    

    def __get_probs__(self, entries: Iterable[ViterbiEntry]) -> List[float]:
        """
        This method returns the probabilities of each step based on a list of ViterbiEntry.
        
        Args:
        * entries (Iterable[ViterbiEntry]): a list of ViterbiEntry.

        Returns:
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        probs = [0.0] * (len(self.steps) - 2)  # exclude BEGIN and END
        max_log_prob = max([entry.log_prob for entry in entries if not np.isnan(entry.log_prob)], default=-np.inf)

        for entry in entries:
            if entry.last_state is not None:
                if entry.last_state.step_index == 0 or entry.last_state.step_index == len(self.steps) - 1:
                    continue
                probs[entry.last_state.step_index - 1] += np.nan_to_num(np.exp(entry.log_prob - max_log_prob))
        probs /= np.sum(probs)
        return probs
    
    
    def set_if_a_step_is_done(self, step_index: int, is_done: bool) -> None:
        """
        This method removes the entries that have the given step index in their history if the step is done, vice versa.
        
        Args:
        * step_index (int): an integer representing the step index.
        * is_done (bool): a boolean value indicating whether the step is done.
        """
        for k, entry in self.curr_entries.items():
            prev_step_indices = set([state.step_index for state in entry.history])
            if (step_index in prev_step_indices) != is_done:
                del self.curr_entries[k]

    
    def set_current_step(self, step_index: int) -> None:
        """
        This method sets the current step of the tracker.

        Args:
        * step_index (int): an integer representing the step index.
        """
        if step_index not in self.curr_entries:
            # print(f'[warning] The step index {step_index} is not in the current entries.')
            best_entry = sorted(self.curr_entries.values(), key=lambda entry: entry.log_prob, reverse=True)[0]
            best_entry.history[-1] = HiddenState(step_index, 0)
            self.curr_entries[step_index] = ViterbiEntry(np.log(1.0), best_entry.history)
        else:
            self.curr_entries[step_index].log_prob = np.log(1.0)


    def forward(self, observation: List[float]) -> Tuple[List[int], List[float]]:
        """
        This method calculates the Viterbi forward algorithm for a single frame given the current prediction entries.

        Args:
        * observation (List[float]): a list of the observation probabilities of each step at the current frame.

        Returns:
        * steps (List[int]): a list of integers representing the step indices in the best entry's history.
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        assert len(self.curr_entries) > 0, 'No current entries to forward. Maybe the tracker is not initilized.'
        assert len(observation) == len(self.steps) - 2, 'The length of the observation probabilities does not match the number of steps.'

        # calibrate frame-based observation probabilities with a known confusion matrix
        confusion_matrix = self._get_context_aware_confusion_matrix()
        observed_log_probs: Dict[int, float] = {}  # dict[step_index, log_prob]
        for actual_step in self.steps: 
            if actual_step.index == 0 or actual_step.index == len(self.steps) - 1:  # BEGIN or END
                observed_log_probs[actual_step.index] = np.log(sys.float_info.epsilon)
                continue
            total_prob = sys.float_info.epsilon  # avoid -np.inf
            for observed_step, cm_prob in zip(self.steps[1: -1], confusion_matrix[actual_step.index - 1]):  # exclude BEGIN and END
                total_prob += cm_prob * observation[observed_step.index - 1]  # exclude BEGIN
            observed_log_probs[actual_step.index] = np.log(total_prob)

        # forward with the observation probabilities
        next_entries: Dict[int, ViterbiEntry] = {}  # dict[step_index, entry]
        for curr_step_index, curr_entry in self.curr_entries.items():
            if curr_entry.log_prob == -np.inf or np.isnan(curr_entry.log_prob):  # discard invalid entries
                continue
            transitions = self._get_context_aware_transitions(curr_entry)  # List[HiddenTransition], length = max_time

            for transition in transitions:  # stay or transit hypotheses     
                log_prob = curr_entry.log_prob + transition.log_prob + observed_log_probs[transition.next_step_index]
                if transition.next_step_index in next_entries and log_prob < next_entries[transition.next_step_index].log_prob:  # already existing better hypothesis
                    continue

                if curr_step_index == transition.next_step_index:  # stay on the step
                    next_state = HiddenState(transition.next_step_index, curr_entry.last_state.time + 1)
                else:
                    next_state = HiddenState(transition.next_step_index, 0)

                next_entries[transition.next_step_index] = ViterbiEntry(log_prob, curr_entry.history + [next_state])
        
        self.curr_entries = next_entries
        self.curr_time += 1
        return self.__get_best_history__(self.curr_entries.values()), self.__get_probs__(self.curr_entries.values())


    def predict_batch(self, observations: List[List[float]], oracle: Optional[Dict[int, int]] = None) -> Iterator[Tuple[List[int], List[float]]]:
        """
        This function is used for predicting the steps of a procedure using the complete observation data.

        Args:
        * observations (List[List[float]]): a numpy array containing the observation probabilities (frames x steps).
        * oracle (Optional[Dict[int, int]]): an optional dictionary where the keys are time and the values are the given step index.

        For each time frame, returns:
        * steps (List[int]): a list of integers representing the step indices in the best entry's history.
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        oracle = {} if oracle is None else oracle

        # dp: basic viterbi algorithm
        for i, observation in enumerate(observations):
            yield self.forward(observation)
            if self.curr_time in oracle:
                self.set_current_step(oracle[self.curr_time])