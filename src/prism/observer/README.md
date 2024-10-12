# PrISM --- Observer module

This is an intervention policy algorithm to enable proactive intervention from the assistant.

# Scripts

## calculate_remaining_time.py

This will generate the estimated remaining time `D_t` distribution over time at `datadrive / tasks / {task_name} / observer / lopo / {pid} / dt_distribution.pkl`.

```
$ python calculate_remaining_time.py --task latte_making
```

You can specify test participants by using `--test_pids`.

Each pickle file contains a dictionary with keys `expectations`, `entropies`, and `ground_truth`.

## evaluate_policy.py

This will generate the results for the intervention policy at `datadrive / tasks / {task_name} / observer / lopo / {pid} / step_threshold_dict.pkl`.

```
$ python evaluate_policy.py --task latte_making
```

You can specify test participants by using `--test_pids`.

# API

```
from prism.observer import ObserverAPI

# target_step: Step 1
policy_config = {
    target_step: {'h_threshold': 0.3, 'offset': 15}
}
observer_api = ObserverAPI(task_name='latte_making')
```
