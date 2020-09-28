import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from typing import Tuple

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger

from model import RecognitionModule
from dataset import VoiceDataModule


class SpeechRecognitionTrainer(pl.LightningModule):
    def __init__(self, learning_rate: float, num_mels: int):
        super(SpeechRecognitionTrainer, self).__init__()

        self.model = RecognitionModule(num_features=num_mels)
        self.criterion = nn.CTCLoss()
        self._learning_rate = learning_rate

    def forward(
            self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.model(x, hidden)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), self._learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return [optimizer], [scheduler]

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        hidden = self.model.get_init_hidden(spectrograms.shape[0], self.device)
        output, _ = self(spectrograms, hidden)
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        res = pl.TrainResult(loss)
        res.log_dict({'train/loss': loss},
                     prog_bar=True, logger=True, on_step=True)
        return res

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        res = pl.EvalResult(checkpoint_on=loss)
        res.log_dict({"val/loss": loss}, prog_bar=True, logger=True, on_epoch=True)
        return res


def main():
    path_csv_dataset = ""
    lr = 1e-3
    datamodule = VoiceDataModule(4, path_csv_dataset, 128)
    log_path = ""
    speech_module = SpeechRecognitionTrainer(lr, 128)

    logger = TensorBoardLogger(log_path, name="speech")
    lr_logger = LearningRateLogger(logging_interval='step')
    trainer = pl.Trainer(
        max_epochs=3, gpus=0, distributed_backend=None,
        logger=logger,
        callbacks=[lr_logger]
    )

    trainer.fit(speech_module, datamodule=datamodule)


if __name__ == '__main__':
    main()
