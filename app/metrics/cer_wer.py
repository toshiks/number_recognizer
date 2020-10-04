import editdistance
import numpy as np

from pytorch_lightning.metrics import NumpyMetric

from dataset.utils import get_index_to_letter_map


class CerWer(NumpyMetric):
    def __init__(self):
        super(CerWer, self).__init__("CerWer")

        self._index_to_letter_map = get_index_to_letter_map()

    def forward(self, predicts: np.ndarray, targets: np.ndarray) -> np.ndarray:

        cer = []
        wer = []

        for predict, target in zip(predicts, targets):
            predict_string = self._index_to_vector(predict)
            target_string = self._index_to_vector(target)

            predict_words = predict_string.rstrip().split(' ')
            target_words = target_string.rstrip().split(' ')

            dist = editdistance.eval(target_string, predict_string)
            dist_word = editdistance.eval(target_words, predict_words)
            cer.append(dist / float(len(target_string)))
            wer.append(dist_word / float(len(predict_words)))

        return np.array([cer, wer])

    def _index_to_vector(self, index_vector):
        return "".join([self._index_to_letter_map[i] for i in index_vector])
