import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from typing import Tuple

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint

from model import RecognitionModule
from dataset import VoiceDataModule


class SpeechRecognitionTrainer(pl.LightningModule):
    def __init__(self, learning_rate: float, num_mels: int):
        super(SpeechRecognitionTrainer, self).__init__()

        self.model = RecognitionModule(num_features=num_mels, out_classes=35)
        self.criterion = nn.CTCLoss(blank=34)
        self._learning_rate = learning_rate

    def forward(
            self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.model(x, hidden)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), self._learning_rate)
        return [optimizer]

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
        res.log_dict({'loss_train': loss}, logger=True, on_step=True)
        return res

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        res = pl.EvalResult(checkpoint_on=loss)
        res.log_dict({"loss_val": loss}, prog_bar=True, logger=True, on_epoch=True)
        return res


def train_model(cfg: DictConfig):
    data_module = VoiceDataModule(
        batch_size=cfg.train_config.batch_size,
        path_csv_dataset=cfg.paths.csv_train,
        num_mels=cfg.train_config.n_mels
    )

    speech_module = SpeechRecognitionTrainer(
        learning_rate=cfg.train_config.lr,
        num_mels=cfg.train_config.n_mels
    )

    logger = TensorBoardLogger(
        save_dir=cfg.paths.log_path,
        name="speech_recognition"
    )

    model_checkpoints = ModelCheckpoint(
        filepath=cfg.paths.save_checkpoints,
        prefix="spr_"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train_config.max_epoches,
        distributed_backend=None,
        gpus=1,
        logger=logger,
        checkpoint_callback=model_checkpoints,
        val_check_interval=1.0
    )

    trainer.fit(speech_module, datamodule=data_module)
