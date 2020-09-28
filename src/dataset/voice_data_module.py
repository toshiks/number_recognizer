import torch
import torch.nn as nn

from functools import partial
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import VoiceDataset
from dataset.preprocessing import Augmentation


class VoiceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, path_csv_dataset: str, num_mels: int = 128):
        super(VoiceDataModule, self).__init__()
        self._batch_size = batch_size
        self._path_csv_dataset = path_csv_dataset
        self._dataset_train, self._dataset_val = None, None
        self._augment = Augmentation()
        self._num_mels = num_mels

    def setup(self, stage: Optional[str] = None):
        dataset = VoiceDataset(path_csv_dataset=self._path_csv_dataset, num_mels=self._num_mels)
        length_dataset = len(dataset)
        train_length_dataset = int(length_dataset * 0.8)
        val_length_dataset = length_dataset - train_length_dataset
        self._dataset_train, self._dataset_val = random_split(
            dataset=dataset,
            lengths=[train_length_dataset, val_length_dataset],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_train, batch_size=self._batch_size, pin_memory=True,
                          collate_fn=self._collate_sequences)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_val, batch_size=self._batch_size, pin_memory=True,
                          collate_fn=partial(self._collate_sequences, is_val=True))

    def _collate_sequences(self, batch, is_val: bool = False):
        spectrograms = []
        labels = []
        length_spectrograms = []
        length_labels = []

        for (spectrogram, label, length_spectrogram, length_label) in batch:
            if not is_val:
                spectrogram = self._augment(spectrogram)
            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))  # time, feature
            labels.append(label)
            length_spectrograms.append(length_spectrogram)
            length_labels.append(length_label)

        spectrograms = nn.utils.rnn.pad_sequence(sequences=spectrograms, batch_first=True).transpose(1, 2)
        labels = nn.utils.rnn.pad_sequence(sequences=labels, batch_first=True)

        return spectrograms, labels, length_spectrograms, length_labels
