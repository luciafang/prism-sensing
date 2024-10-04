from typing import Dict, List, Optional


class Step:
    def __init__(self, index: int, description: str, mean_time: float, std_time: float):
        self.index = index
        self.description = description
        self.mean_time = mean_time
        self.std_time = std_time

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Step):
            # don't attempt to compare against unrelated types
            return False
        return self.index == __value.index
    
    def __hash__(self) -> int:
        return id(self.index)
    
    def __repr__(self):
        return f's{self.index}--{self.description}'


class History:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, History):
            # don't attempt to compare against unrelated types
            return False
        if not len(self.steps) == len(__value.steps):
            return False
        return all([self.steps[i] == __value.steps[i] for i in range(len(self.steps))])
    
    def __hash__(self) -> int:
        return id([step.index for step in self.steps])
    
    def __repr__(self):
        return f'{[step.index for step in self.steps]}'


class Graph:
    def __init__(self, steps: List[Step], edges: Dict[Step, Dict[Step, float]], histories: List[History]):
        self.steps = steps
        self.edges = edges
        self.start = self.steps[0]
        self.end = self.steps[-1]

        self.history_probs = {}
        for history in histories:
            if history not in self.history_probs.keys():
                self.history_probs[history] = 1/len(histories)
            else:
                self.history_probs[history] += 1/len(histories)

    def __repr__(self) -> str:
        text = f'Graph: {len(self.steps)} steps, {len(self.edges)} edges, start={self.start}, end={self.end}\n'
        for step in self.steps:
            text += f'{step} ({round(step.mean_time, 1)} +- {round(step.std_time, 1)}) -> {dict(map(lambda x: (x[0], round(x[1], 1)), self.edges[step].items()))}\n'
        return text
    
    def get_potential_histories(self, partial_history: History) -> Dict[History, float]:
        potential_histories = {}
        for history, prob in self.history_probs.items():
            if len(history.steps) < len(partial_history.steps):
                continue
            if all([history.steps[i] == partial_history.steps[i] for i in range(len(partial_history.steps))]):
                potential_histories[history] = prob
        return potential_histories

    def get_potential_next_steps(self, step: Step) -> List[Step]:
        return list(self.edges[step].keys())


class HiddenState:
    def __init__(self, step_index: int, time: int):
        self.step_index = step_index
        self.time = time

    def __repr__(self):
        return f'{self.step_index}_{self.time}'


class HiddenTransition:
    def __init__(self, next_step_index: int, log_prob: float):
        self.next_step_index = next_step_index
        self.log_prob = log_prob

    def __repr__(self):
        return f'{self.next_step_index=}@{self.log_prob}'


class ViterbiEntry:
    def __init__(self, log_prob: float, history: List[HiddenState]):
        self.log_prob = log_prob
        self.history = history

    def __repr__(self):
        text = ''
        prev_state = None
        for state in self.history:
            if prev_state is None:
                prev_state = state
            if state.step_index != prev_state.step_index:
                text += f'{prev_state}->'
            prev_state = state
        text += f'{state}'
        return f'{text}@{self.log_prob}'

    @property
    def last_state(self) -> Optional[HiddenState]:
        if len(self.history) == 0:
            return None
        return self.history[-1]