# PrISM --- HAR module

This is a frame-level activity recognition. 
The backbone code comes from [SAMoSA](https://github.com/cmusmashlab/SAMoSA/tree/main).

# Scripts

## featurization.py

This will create frame-level multimodal features at `datadrive / tasks / {task_name} / dataset / featurized` by using data at `datadrive / tasks / {task_name} / dataset / original`. 

```
$ python featurization.py --task latte_making
```

You can apply the featurization to a specific participant by using `--test_pids`.

## classificattion.py

This will generate results for the frame-level step classification (Leave-One-Participant-Out) at `datadrive / tasks / {task_name} / models`.

```
$ python classification.py --task latte_making
```

You can specify test participants by using `--test_pids`.

# API

```
from prism.har import HumanActivityRecognitionAPI

har_api = HumanActivityRecognitionAPI(task_name='latte_making')
```
