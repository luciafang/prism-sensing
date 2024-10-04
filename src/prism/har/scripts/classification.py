"""
This module contains functions for training a classifier and obtaining confusion probabilities.

after
- datadrive / tasks / {task_name} / models / lopo / {pid} / model.pkl
- datadrive / tasks / {task_name} / models / lopo / {pid} / true.pkl
- datadrive / tasks / {task_name} / models / lopo / {pid} / pred_raw.pkl
- datadrive / tasks / {task_name} / models / lopo / {pid} / cm_raw.pkl
- datadrive / tasks / {task_name} / models / lopo / {pid} / cm_raw.png
- datadrive / tasks / {task_name} / models / lopo / {pid} / result_raw.txt
- datadrive / tasks / {task_name} / models / lopo / cm_raw.pkl
- datadrive / tasks / {task_name} / models / lopo / cm_raw.png
- datadrive / tasks / {task_name} / models / lopo / result_raw.txt
- datadrive / tasks / {task_name} / models / all / model.pkl
"""

import argparse
import pathlib
import pickle
from typing import Union, List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


from prism import config
from prism.har.algorithm import Classifier, Evaluator


def load_imu_and_audio_data(pickle_files: List[Union[str, pathlib.Path]],
                            steps: List[str]) -> Tuple[npt.NDArray, List[int]]:
    """
    This function loads IMU and audio data from a set of pickle files and converts the labels into numerical values based on their index in a list of steps.

    Args:
    * pickle_files (List[Union[str, pathlib.Path]]): a list of paths to the pickle files containing IMU and audio data.
    * steps (List[str]): a list of strings representing the different steps in the procedure.

    Returns:
    * X (npt.NDArray): a 2D numpy array containing the frame-based time-series IMU and audio data.
    * y (List[int]): a list of integers representing the index of the step for each time frame.
    """
    X, y = None, []
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as fp:
            data = pickle.load(fp)
        if 'DUMMY' in data['label']:
            print(f'Warning: DUMMY label found in {pickle_file}. Skipping...')
            continue
            
        motion_data, audio_data, labels = [], [], []
        for i, label in enumerate(data['label']):
            if label != 'OTHER':
                motion_data.append(data['motion'][i])
                audio_data.append(data['audio'][i])
                labels.append(label)
            else:
                print('Warning: OTHER label found in the data. Skipping...')

        x = np.hstack((np.array(motion_data), np.array(audio_data)))
        if X is None:
            X = x
        else:
            X = np.vstack((X, x))
        
        label_indices = list(map(lambda l: steps.index(l), labels))
        y += label_indices
    return X, y


def get_args():
    # TODO: add mode for lopo / inference-only / all
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--test_pids', type=str, help='To specify test pids (e.g., --test_pids 10,11,14)', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    working_dir = task_dir / 'models'
    with open(task_dir / 'dataset' / 'steps.txt', 'r') as f:
        steps = [s.strip() for s in f.readlines()]

    feature_files = list(task_dir.glob('dataset/featurized/*.pkl'))
    if args.test_pids is not None:
        test_files = [f for f in feature_files if f.stem in args.test_pids.split(',')]
    else:
        test_files = feature_files
    print('Test PIDs:', [f.stem for f in test_files])
    y_test_concatenated, y_pred_concatenated = [], []
    evaluator = Evaluator()

    for test_file in test_files:
        print(f'Leave-one-participant-out: {test_file.stem}')
        pid_dir = working_dir / 'lopo' / test_file.stem
        pid_dir.mkdir(parents=True, exist_ok=True)
                                                      
        train_files = [f for f in feature_files if f != test_file]
        X_train, y_train = load_imu_and_audio_data(train_files, steps)
        X_test, y_test = load_imu_and_audio_data([test_file], steps)
        for class_id in range(len(steps)):  # add dummy data for classes not appeared
            if class_id not in y_train:
                X_train = np.vstack((X_train, np.zeros((1, X_train.shape[1]))))
                y_train = y_train + [class_id]

        clf = Classifier()
        clf.train(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)
        clf.save(pid_dir / 'model.pkl')
        with open(pid_dir / 'true.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        with open(pid_dir / 'pred_raw.pkl', 'wb') as f:
            pickle.dump(y_pred_proba, f)
        accuracy, f1 = evaluator.frame_level_metrics(y_test, np.argmax(y_pred_proba, axis=1))
        with open(pid_dir / 'result_raw.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1: {f1}\n')
        cm = evaluator.confusion_matrix(y_test, np.argmax(y_pred_proba, axis=1), labels=range(len(steps)))
        cm /= cm.sum(axis=1, keepdims=True)
        with open(pid_dir / 'cm_raw.pkl', 'wb') as f:
            pickle.dump(cm, f)
        plt.imshow(cm, cmap='Blues')
        plt.savefig(pid_dir / 'cm_raw.png')
        plt.close()
        y_test_concatenated = np.concatenate((y_test_concatenated, y_test))
        y_pred_concatenated = np.concatenate((y_pred_concatenated, np.argmax(y_pred_proba, axis=1)))

    # aggregate results
    if len(y_test_concatenated) > 0:
        accuracy, f1 = evaluator.frame_level_metrics(y_test_concatenated, y_pred_concatenated)
        with open(working_dir / 'lopo' / 'result_raw.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1: {f1}\n')
        cm = evaluator.confusion_matrix(y_test_concatenated, y_pred_concatenated, labels=range(len(steps)))
        cm /= cm.sum(axis=1, keepdims=True)
        with open(working_dir / 'lopo' / 'cm_raw.pkl', 'wb') as f:
            pickle.dump(cm, f)
        plt.imshow(cm, cmap='Blues')
        plt.savefig(working_dir / 'lopo' / 'cm_raw.png')
        plt.close()
        
    # train on all data
    print('Train on all data')
    (working_dir / 'all').mkdir(parents=True, exist_ok=True)
    X_all, y_all = load_imu_and_audio_data(feature_files, steps)
    clf = Classifier()
    clf.train(X_all, y_all)
    clf.save(working_dir / 'all' / 'model.pkl')