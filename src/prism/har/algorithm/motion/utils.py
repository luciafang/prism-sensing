import numpy as np

from ... import params


def get_motion_examples(motion_data):
    """
    Get motion data by windows.

    Args:
        motion_data (np.ndarray): motion data (n_samples, n_features)

    Returns:
        np.ndarray: motion examples (n_frames, window_length, n_features)
    """
    window_length = params.WINDOW_LENGTH_IMU
    hop_length = params.HOP_LENGTH_IMU

    if motion_data.shape[0] < window_length:  # zero padding
        len_pad = int(np.ceil(window_length)) - motion_data.shape[0]
        to_pad = np.zeros((len_pad, ) + motion_data.shape[1:])
        motion_data = np.concatenate([motion_data, to_pad], axis=0) 

    num_samples = motion_data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, int(window_length)) + motion_data.shape[1:]
    strides = (motion_data.strides[0] * int(hop_length),) + motion_data.strides
    return np.lib.stride_tricks.as_strided(motion_data, shape=shape, strides=strides)


def normalize_motion(motion, norm_params):
    pseudo_max = norm_params['max']
    pseudo_min = norm_params['min']
    mean = norm_params['mean']
    std = norm_params['std']

    motion_normalized = 1 + (motion - pseudo_max) * 2 / (pseudo_max - pseudo_min)
    motion_normalized = (motion_normalized - mean) / std
    return motion_normalized