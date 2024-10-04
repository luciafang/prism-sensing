from typing import List, Optional, Tuple

import numpy as np
import matplotlib.axes
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)


class Evaluator():

    def confusion_matrix(self, y_true, y_pred, labels=None):
        return confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    
    def frame_level_metrics(self, y_true: List[int], y_pred: List[int], num_classes: Optional[int] = None,
                            ax: Optional[matplotlib.axes.Axes] = None, verbose: bool = False) -> Tuple[float, float]:
        """
        This function computes time frame-level metrics for a given set of true and predicted labels.
        The metrics include accuracy, recall, precision, and F1 score for all classes combined, which are returned as a tuple.
        If a matplotlib axes object is provided, a confusion matrix is also plotted.

        Args:
        * y_true_series (List[int]): a list of true labels.
        * y_pred_series (List[int]): a list of predicted labels.
        * num_classes (Optional(int)): the number of classes, i.e., the number of steps in a procedure.
        * ax (Optional[matplotlib.axes.Axes]): a matplotlib axes object to plot the confusion matrix on.
        * verbose (bool): a flag whether to print the computed metrics.

        Returns:
        * all_accuracy (float): a float value of the overall accuracy score.
        * all_f1 (float): a float value of the overall macro F1 score.
        """

        if ax is not None:
            if num_classes is None:
                num_classes = len(np.unique(y_true))
            cm = confusion_matrix(y_true, y_pred, labels=range(num_classes), normalize='true')
            d = ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues', colorbar=False, values_format='.2g',
                                                im_kw={'vmin': 0, 'vmax': 1})

            for text in d.text_.flatten():
                text.set_clip_on(True)

            ax.set_xlim(0.5, num_classes - 1.5)
            ax.set_ylim(num_classes - 1.5, 0.5)

            ax.xaxis.label.set_fontsize(16)
            ax.yaxis.label.set_fontsize(16)
            ax.tick_params(axis='both', which='major', labelsize=14)

        all_accuracy = accuracy_score(y_true, y_pred)
        all_recall = recall_score(y_true, y_pred, average='macro')
        all_precision = precision_score(y_true, y_pred, average='macro')
        all_f1 = f1_score(y_true, y_pred, average='macro')

        if verbose:
            print('Overall accuracies:', all_accuracy)
            print('Overall macro recall:', all_recall)
            print('Overall macro precision:', all_precision)
            print('Overall macro F1:', all_f1)

        return all_accuracy, all_f1
