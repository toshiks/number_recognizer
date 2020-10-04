import torch
import torch.nn as nn

from typing import Tuple
from model.layers import InputConv1d, DenseDropoutBlock, NoLinearityBlock


class RecognitionModule(nn.Module):
    def __init__(self, num_features: int, hidden_dense_size: int = 128,
                 hidden_lstm_size: int = 1024, out_classes: int = 35):
        super(RecognitionModule, self).__init__()
        self._num_lstm_layers = 1
        self._hidden_lstm_size = hidden_lstm_size
        self._cnn = InputConv1d(num_features)
        self._dense = nn.Sequential(
            DenseDropoutBlock(num_features, hidden_dense_size),
            DenseDropoutBlock(hidden_dense_size, hidden_dense_size),
        )
        self._lstm = nn.LSTM(
            input_size=hidden_dense_size,
            hidden_size=hidden_lstm_size,
            num_layers=self._num_lstm_layers
        )
        self._no_linearity_lstm = NoLinearityBlock(hidden_lstm_size)
        self._out_dense = nn.Linear(hidden_lstm_size, out_classes)

    def forward(
            self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: batch, feature, time

        x = self._cnn(x)  # batch, time, feature
        x = self._dense(x)  # batch, time, feature
        out, (hn, cn) = self._lstm(x.transpose(0, 1), hidden)  # time, batch, feature
        out = self._no_linearity_lstm(out)

        return self._out_dense(out), (hn, cn)

    def get_init_hidden(self, batch_size, device):
        return (torch.zeros(self._num_lstm_layers, batch_size, self._hidden_lstm_size, device=device),
                torch.zeros(self._num_lstm_layers, batch_size, self._hidden_lstm_size, device=device))
