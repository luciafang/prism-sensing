import numpy as np


from .algorithm import FeatureExtractor
from .algorithm import Classifier
from .. import config

class HumanActivityRecognitionAPI():
    """
    API for Human Activity Recognition, mainly for the streaming purpose.
    """

    def __init__(self, task_name):
        clf_path = config.datadrive / 'tasks' / task_name / 'models' / 'all' / 'model.pkl'
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()
        self.classifier.load(clf_path)

    def __call__(self, data) -> list:
        """
        Predicts the activity label for the given data.
        If the audio and motion data have different lengths, only the last "windowed example" is used.

        Args:
        * data (dict): a dictionary containing the audio and motion data.
            audio should not be normalized.

        Returns:
        * list: a list of probability for each activity label.
        """
        audio_feature = self.feature_extractor.featurize_data(data['audio'], dtype='audio')
        motion_feature = self.feature_extractor.featurize_data(data['motion'], dtype='motion')
        if audio_feature.shape[0] != motion_feature.shape[0]:
            audio_feature = audio_feature[-1:]
            motion_feature = motion_feature[-1:]
        feature = np.hstack((motion_feature, audio_feature))  # order is important
        return self.classifier.predict_proba(feature)[0]