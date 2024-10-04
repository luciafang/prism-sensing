from . import mel_features
from ... import params


def get_audio_examples(audio_data):
    """
    Get audio data by windows.

    Args:
        audio_data (np.ndarray): audio data (n_samples,)
    
    Returns:
        np.ndarray: audio examples (n_frames, n_window_samples, num_mel_bins)
    """
    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(audio_data,
                                               audio_sample_rate=params.SAMPLE_RATE,
                                               log_offset=params.LOG_OFFSET,
                                               window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                               hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                               num_mel_bins=params.NUM_MEL_BINS,
                                               lower_edge_hertz=10,
                                               upper_edge_hertz=params.SAMPLE_RATE // 2)

    # Frame features into examples.
    example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS / params.STFT_HOP_LENGTH_SECONDS))  # 96
    example_hop_length = int(round(params.EXAMPLE_HOP_SECONDS / params.STFT_HOP_LENGTH_SECONDS))  # 7
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length
    )
    return log_mel_examples
