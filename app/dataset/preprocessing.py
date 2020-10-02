import numpy as np

import torch
import torch.nn as nn
import torchaudio
import num2words as n2w

from dataset.utils import get_index_to_letter_map, get_letter_to_index_map


class LogMelSpectrogram(nn.Module):
    """
    Create spectrogram from raw audio and make
    that logarithmic for avoiding inf values.
    """
    def __init__(self, sample_rate: int = 8000, n_mels: int = 128):
        super(LogMelSpectrogram, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transform(waveform)
        return torch.log(spectrogram + 1e-10)


class Augmentation:
    def __init__(self):
        self.transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=15)
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transforms(waveform)


class NumberToTextVec:
    """
    Convert number to to feature vector
    """
    def __init__(self):
        self._letter_to_index = get_letter_to_index_map()
        self._index_to_letter = get_index_to_letter_map()

    @staticmethod
    def number_to_text(number: int) -> str:
        return n2w.num2words(number, lang='ru')

    def text_to_index_vector(self, text: str) -> np.array:
        vector = []
        for i in text:
            vector.append(self._letter_to_index[i])

        return np.array(vector)

    def number_to_index_vector(self, number: int) -> np.array:
        return self.text_to_index_vector(NumberToTextVec.number_to_text(number))

    def index_vector_to_text(self, index_vector: np.array):

        return "".join([self._index_to_letter[i] for i in index_vector])
