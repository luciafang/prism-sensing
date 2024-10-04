"""
Featurization script for the HAR task.
before
- datadrive / tasks / {task_name} / dataset / original

after
- datadrive / tasks / {task_name} / dataset / featurized
"""
import argparse
import numpy as np
import pandas as pd
import pickle
from scipy.io import wavfile

from prism import config
from prism.har import params
from prism.har.algorithm import FeatureExtractor
from prism.har.algorithm.audio import get_audio_examples
from prism.har.algorithm.motion import get_motion_examples


def get_label(t, times, steps):
    for i, time in enumerate(times):
        if t < time:
            return steps[i - 1]
    raise ValueError(f'No label found for {t=}') 


def remove_unnecessary_steps_in_the_beginning_and_end(audio_examples, motion_examples, labels, times):
    """
    Delete beginning and end examples classified as 'OTHER'.
    """
    other_categories = ['BEGIN', 'OTHER', 'END']  # TODO: this is hard-coded
    i = 0
    last_index = len(labels) - 1
    j = last_index
    while (i < len(labels)):
        if str(labels[i]) in other_categories:
            i += 1
        else:
            break
    while (j >= 0):
        if str(labels[j]) in other_categories:
            j -= 1
        else:
            break

    audio = audio_examples[i:j + 1,]
    imu = motion_examples[i:j + 1,]
    labels = labels[i:j + 1]
    times = times[i:j + 1]
    print(f'removed other at the beginning and ending: 0 -- {i}, {j} -- {last_index}', audio.shape, imu.shape, len(labels), len(times))
    assert len(labels) == audio.shape[0] == imu.shape[0] == len(times)
    return audio, imu, labels, times


def create_feature_pkl(pid, feature_extractor):
    pid_dir = task_dir / 'dataset' / 'original' / pid
    annotation_path = pid_dir / 'annotation.txt'
    if not annotation_path.exists():
        annotation_times = [1e9]
        annotation_steps = ['DUMMY']
    else:
        annotation = pd.read_csv(pid_dir / 'annotation.txt')
        annotation = annotation.sort_values(by='Timestamp')
        annotation_times, annotation_steps = annotation['Timestamp'].tolist(), annotation['Step'].tolist()
    # load data
    audio_file_path = pid_dir / 'audio.wav'
    motion_file_path = pid_dir / 'motion.txt'

    # get examples
    ## audio (every 0.21 sec)
    sr, audio_data = wavfile.read(audio_file_path)  # audio data (n_samples, n_channels))
    assert audio_data.dtype == 'int16' and sr == params.SAMPLE_RATE, 'Invalid audio file format.'
    audio_data = audio_data / (2**15)        # Convert signed 16-bit to [-1.0, +1.0]    # Convert to [-1.0, +1.0]
    if len(audio_data.shape) > 1:      # Convert to mono.
        audio_data = np.mean(audio_data, axis=1)
    audio_examples = get_audio_examples(audio_data)  # audio data (n_frames, n_window_samples, num_mel_bins)   
 
    ## motion (every 0.2 sec)
    motion_df = pd.read_csv(motion_file_path, sep='\s+', header=0, engine='python')
    motion_data = motion_df.to_numpy()[:, 1:]  # motion data (n_samples, n_features)
    motion_examples = get_motion_examples(motion_data)  # motion data (n_frames, window_length, n_features)
    motion_timestamps = motion_df['timestamp'].tolist()
    print(f'Loaded data for {pid}: {audio_examples.shape=}, {motion_examples.shape=}')

    # align motion and audio
    aligned_audio_examples = []
    aligned_motion_examples = []
    labels = []
    relative_times = []

    # loop through all the audio examples and
    for i in range(audio_examples.shape[0]):
        end_audio_sec = params.EXAMPLE_WINDOW_SECONDS + params.EXAMPLE_HOP_SECONDS * i
        motion_sample_num = params.SAMPLE_RATE_IMU * end_audio_sec
        motion_example_index = int((motion_sample_num - params.WINDOW_LENGTH_IMU) / params.HOP_LENGTH_IMU)
        if motion_example_index >= motion_examples.shape[0]:
            print(f'out of bounds {motion_example_index=} {motion_examples.shape[0]=} {i=} {audio_examples.shape[0]=}')
            break
        motion_example = motion_examples[motion_example_index, :, :]

        # get the timestamp from the motion data
        last_frame_index_motion_example = int(motion_example_index * params.HOP_LENGTH_IMU + params.WINDOW_LENGTH_IMU)
        ms = motion_timestamps[last_frame_index_motion_example]

        if ms > annotation_times[-1]:
            print(f'Break at {ms=} > {annotation_times[-1]=}')
            break

        try:
            label = get_label(ms, annotation_times, annotation_steps)
            labels.append(label)
            relative_times.append(end_audio_sec * 1000)
            aligned_motion_examples.append(motion_example)
            aligned_audio_examples.append(audio_examples[i, :, :])
        except Exception as e:
            print(f'Error in getting label for {pid}: ', e)
            continue

    aligned_audio_examples = np.array(aligned_audio_examples)
    aligned_motion_examples = np.array(aligned_motion_examples)
    assert len(labels) == len(relative_times) == aligned_motion_examples.shape[0] == aligned_audio_examples.shape[0]

    print(f'Featurizing for {pid}: {aligned_audio_examples.shape=}, {aligned_motion_examples.shape=}, {len(labels)=}, {len(relative_times)=}')

    audio_examples, motion_examples, labels, new_times = remove_unnecessary_steps_in_the_beginning_and_end(
                                                            aligned_audio_examples,
                                                            aligned_motion_examples,
                                                            labels,
                                                            relative_times
                                                        )

    dataset = {
        'motion': feature_extractor.featurize_examples(motion_examples, dtype='motion'),
        'audio': feature_extractor.featurize_examples(audio_examples, dtype='audio'),
        'label': labels,
        'timestamp': new_times
    }

    print(f'Featurized done for {pid}: {dataset["motion"].shape=}, {dataset["audio"].shape=}, {len(dataset["label"])=}, {len(dataset["timestamp"])=}')
    return dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--test_pids', type=str, help='To specify test pids (e.g., --test_pids 10,11,14)', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    feature_extractor = FeatureExtractor()
    to_be_featurized_pids = [f.stem for f in task_dir.glob('dataset/original/*') if f.is_dir()]
    if args.test_pids is not None:
        to_be_featurized_pids = [pid for pid in to_be_featurized_pids if pid in args.test_pids.split(',')]
    print('PIDs to be featurized:', to_be_featurized_pids)

    for pid in to_be_featurized_pids:
        print(f'-----Create feature pkl for {pid}-----')
        dataset = create_feature_pkl(pid, feature_extractor)
        save_fp = task_dir / 'dataset' / 'featurized' / f'{pid}.pkl'
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(save_fp, 'wb') as f:
            pickle.dump(dataset, f)
