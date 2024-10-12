import numpy as np

class InterventionPolicy():

    def __init__(self, target_step, h_threshold, offset=0, h_average_window=10, e_stability_window=50, e_stability_threshold=150):
        """
        Args:
        * target_step (Step): the target step for the intervention policy.
        * h_threshold (float): the threshold for the entropy value.
        * offset (int): an offset value for the triggered timer. + values for `notify if forgotten` and - values for `remind in advance`.
        * h_average_window (int): a window size for the entropy average.
        * e_stability_window (int): a window size for the stability check.
        * e_stability_threshold (int): a threshold for the stability check.
        """
        self.target_step = target_step  # Step
        self.h_threshold = h_threshold
        self.offset = offset
        self.h_average_window = h_average_window
        self.e_stability_window = e_stability_window
        self.e_stability_threshold = e_stability_threshold
        self.reset()
        
    def reset(self):
        """
        Reset the internal state.
        """
        self.expectations = []
        self.entropys = []
        self.time = 0
        self.is_timer_started = False
        self.timer_start_time = None
        self.original_timer_duration = None

    def forward(self, e, h):
        """
        Args:
        * e (float): the expected remaining time for the target step.
        * h (float): the entropy of the expected remaining time for the target step.

        Returns:
        * status (str): a status of the intervention policy. 'timer_start', 'timer_stop', or 'no_action'.
        * timer_duration (Optional(int)): a duration of the timer in frames. Only available when the status is 'timer_start'.
        """
        self.expectations.append(e)
        self.entropys.append(h)
        self.time += 1
        if self.is_timer_started:
            elapsed_time = self.time - self.timer_start_time
            if elapsed_time > self.e_stability_window:
                return 'no_action', None
            else:
                if abs(e - (self.original_timer_duration - elapsed_time)) > self.e_stability_threshold:
                    self.is_timer_started = False
                    self.timer_start_time = None
                    self.original_timer_duration = None
                    return 'timer_stop', None
                else:
                    return 'no_action', None
        else:
            if len(self.expectations) > self.e_stability_window and sum(self.expectations[-self.e_stability_window:]) == 0:
                ## the step already happened
                self.is_timer_started = True
                self.timer_start_time = self.time
                self.original_timer_duration = 0
                return 'timer_start', self.original_timer_duration
            
            averaged_h = np.mean(self.entropys[-self.h_average_window:])
            if averaged_h > self.h_threshold:
                return 'no_action', None
            else:
                self.is_timer_started = True
                self.timer_start_time = self.time
                self.original_timer_duration = max(e + self.offset, 0)
                return 'timer_start', self.original_timer_duration
    
    def predict_batch(self, expectations, entropys):
        """
        Args:
        * expectations (Dict[Step, List[float]]): a dictionary of the expected remaining times for each step.
        * entropys (Dict[Step, List[float]]): a dictionary of the entropys of the expected remaining time for each step.

        Returns:
        * intervention_time (int): a time when the intervention happens.
        """
        time = 0
        intervention_time = None
        for step, v in expectations.items():
            if step.index == self.target_step.index:
                _expectations = v
                break
        for step, v in entropys.items():
            if step.index == self.target_step.index:
                _entropys = v
                break

        for e, h in zip(_expectations, _entropys):
            time += 1
            status, remaining_time = self.forward(e, h)
            if status == 'timer_start':
                intervention_time = time + remaining_time
            elif status == 'timer_stop':
                intervention_time = None
            else:
                pass
        if intervention_time is None:
            intervention_time = _expectations[1] + 1
        return intervention_time


class BaselinePolicy():

    def __init__(self, target_step):
        self.target_step = target_step  

    def predict_batch(self, expectations):
        """
        Args:
        * expectations (Dict[Step, List[float]]): a dictionary of the expected remaining times for each step.

        Returns:
        * intervention_time (int): a time when the intervention happens.
        """
        for step, v in expectations.items():
            if step.index == self.target_step.index:
                _expectations = v
                break
        return _expectations[1] + 1