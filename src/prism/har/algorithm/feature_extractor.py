import numpy as np
import pickle as pkl
from keras.models import Model, load_model

from prism import config
from prism.har.algorithm.audio import get_audio_examples
from prism.har.algorithm.motion import get_motion_examples, normalize_motion


class FeatureExtractor():


    def __init__(self) -> None:
        self.audio_model = self._build_audio_only_model()
        self.motion_model = self._build_motion_only_model()
        self.motion_norm_params = self._get_motion_norm_params()
        print('FeatureExtractor initialized.')

    def _build_audio_only_model(self):
        path_to_model = config.datadrive / 'pretrained_models' /'audio_model.h5'
        audio_model = load_model(path_to_model)
        fc2_op = audio_model.get_layer('fc2').output
        final_model = Model(
            inputs=audio_model.inputs,
            outputs=fc2_op,
            name='somohar_sound_model'
        )
        return final_model


    def _build_motion_only_model(self):
        path_to_model = config.datadrive / 'pretrained_models' / 'motion_model.h5'
        motion_model = load_model(path_to_model)
        dense2_op = motion_model.get_layer('dense_2').output
        final_model = Model(
            inputs=motion_model.inputs,
            outputs=dense2_op,
            name='somohar_motion_model'
        )
        return final_model


    def _get_motion_norm_params(self):
        path_to_params = config.datadrive / 'pretrained_models' / 'motion_norm_params.pkl'
        with open(path_to_params, 'rb') as f:
            norm_params = pkl.load(f)
        return norm_params
    

    def featurize_examples(self, examples, dtype='audio'):
        if dtype == 'audio':
            return self.audio_model(examples)
        elif dtype == 'motion':
            examples = normalize_motion(examples, self.motion_norm_params)
            return self.motion_model(examples)
        else:
            raise ValueError(f'Invalid dtype: {dtype}')
        
    def featurize_data(self, data, dtype='audio'):
        """
        Args:
        * data: np.ndarray
            if dtype == 'audio', shape = (n_samples, n_channels)
            if dtype == 'motion', shape = (n_samples, n_channels)
        """
        if dtype == 'audio':
            data = data / (2**15)        # Convert signed 16-bit to [-1.0, +1.0]
            if len(data.shape) > 1:      # Convert to mono.
                data = np.mean(data, axis=1)
            examples = get_audio_examples(data)
        elif dtype == 'motion':
            examples = get_motion_examples(data)
        ret = self.featurize_examples(examples, dtype)
        return ret