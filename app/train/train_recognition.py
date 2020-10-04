import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from typing import Tuple

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from app.model import RecognitionModule, CTCDecoder
from app.dataset import VoiceDataModule, get_blank_id, get_alphabet
from app.metrics import CerWer


class SpeechRecognitionTrainer(pl.LightningModule):
    def __init__(self, learning_rate: float, num_mels: int):
        super(SpeechRecognitionTrainer, self).__init__()

        alphabet = list(get_alphabet())
        blank_id = get_blank_id()

        self.model = RecognitionModule(num_features=num_mels, out_classes=len(alphabet))
        self.criterion = nn.CTCLoss(blank=blank_id)
        self._learning_rate = learning_rate
        self._ctc_decoder = CTCDecoder(blank_id=blank_id, alphabet=alphabet)
        self._cer_metric = CerWer()

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
        out_probs = F.softmax(output, dim=2)
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss, out_probs, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        res = pl.TrainResult(loss)
        res.log('loss/train', loss, logger=True, on_step=True, on_epoch=False)
        return res

    def validation_step(self, batch, batch_idx):
        loss, output, labels = self.step(batch)
        output = output.transpose(0, 1)

        decoded_output, _ = self._ctc_decoder(output)
        metrics = self._cer_metric(decoded_output, labels)

        res = pl.EvalResult()

        res.loss_val = loss
        res.cer, res.wer = metrics

        return res

    def validation_epoch_end(self, res):
        res.cer = torch.mean(res.cer)
        res.wer = torch.mean(res.wer)
        res.checkpoint_on = res.wer
        res.loss_val = torch.mean(res.loss_val)
        res.log('metric/cer', res.cer, prog_bar=True, logger=True, on_epoch=True)
        res.log('metric/wer', res.wer, prog_bar=True, logger=True, on_epoch=True)
        res.log('loss/val', res.loss_val, prog_bar=True, logger=True, on_epoch=True)
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
        prefix="spr_",
        save_weights_only=True
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
