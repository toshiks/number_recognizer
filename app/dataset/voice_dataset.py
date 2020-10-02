import os

import torch
import torchaudio
import pandas as pd

from typing import Tuple

from torch.utils.data import Dataset

from dataset.preprocessing import NumberToTextVec, LogMelSpectrogram


class VoiceDataset(Dataset):
    """
    Parsing CSV Dataset, load raw-audio and labels  and preprocessing
    """
    def __init__(self, path_csv_dataset: str, num_mels: int = 128):
        super(VoiceDataset, self).__init__()
        self._path_dataset = os.path.dirname(path_csv_dataset)
        self._data = pd.read_csv(path_csv_dataset)
        self._number_process = NumberToTextVec()
        self._transform = LogMelSpectrogram(n_mels=num_mels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        path, number = self._data[["path", "number"]].iloc[index]
        path_to_file = os.path.join(self._path_dataset, path)

        waveform, _ = torchaudio.load(path_to_file)
        label = self._number_process.number_to_index_vector(number)
        label = torch.tensor(label)

        # channel, feature, time
        spectrogram = self._transform(waveform)

        return spectrogram, label, spectrogram.shape[-1] // 2, len(label)

    def __len__(self):
        return self._data.shape[0]
