# PrISM --- Tracker module

This is an extended Viterbi algorithm to postprocess the frame-level HAR outputs by leveraging a transition graph.

# Scripts

## tracker.py

This will generate post-processed results at `datadrive / tasks / {task_name} / models`.

```
$ python tracker.py --task latte_making
```

You can specify test participants by using `--test_pids`.

# API

```
from prism.tracker import TrackerAPI

tracker_api = TrackerAPI(task_name='latte_making')
```
