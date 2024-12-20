# MFCC Spectrogram conversion code from VGGish, Google Inc.
# https://github.com/tensorflow/models/tree/master/research/audioset

import numpy as np


def frame(data, window_length, hop_length):
    if data.shape[0] < window_length:
        # pad zeros
        len_pad = int(np.ceil(window_length)) - data.shape[0]
        to_pad = np.zeros((len_pad, ) + data.shape[1:])
        data = np.concatenate([data, to_pad], axis=0)
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, int(window_length)) + data.shape[1:]
    strides = (data.strides[0] * int(hop_length),) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))


def stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    frames = frame(signal, window_length, hop_length)
    window = periodic_hann(int(window_length))
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


def hertz_to_mel(frequencies_hertz):
    # Mel spectrum constants and functions.
    _MEL_BREAK_FREQUENCY_HERTZ = 700.0
    _MEL_HIGH_FREQUENCY_Q = 1127.0
    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def get_mel_matrix(audio_sample_rate,
                    num_mel_bins=20,
                    num_spectrogram_bins=129,
                    lower_edge_hertz=125.0,
                    upper_edge_hertz=3800.0):

    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" % (lower_edge_hertz, upper_edge_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel))
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(data, audio_sample_rate, log_offset, window_length_secs, hop_length_secs, **kwargs):
    window_length_samples = audio_sample_rate * window_length_secs
    hop_length_samples = audio_sample_rate * hop_length_secs
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    spectrogram = stft_magnitude(data, fft_length=fft_length, hop_length=hop_length_samples, window_length=window_length_samples)
    mel_matrix = get_mel_matrix(audio_sample_rate, num_spectrogram_bins=spectrogram.shape[1], **kwargs)
    mel_spectrogram = np.dot(spectrogram, mel_matrix)
    return np.log(mel_spectrogram + log_offset)
