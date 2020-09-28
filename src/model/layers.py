import torch
import torch.nn as nn


class NoLinearityBlock(nn.Module):
    def __init__(self, num_features: int):
        super(NoLinearityBlock, self).__init__()
        self._dropout = nn.Dropout(0.1)
        self._norm_layer = nn.LayerNorm(num_features)
        self._gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._norm_layer(x)
        x = self._gelu(x)
        return self._dropout(x)


class InputConv1d(nn.Module):
    def __init__(self, num_features: int):
        super(InputConv1d, self).__init__()
        self._conv = nn.Conv1d(num_features, num_features, 10, 2, 5)
        self._no_linearity = NoLinearityBlock(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        return self._no_linearity(x.transpose(1, 2))


class DenseDropoutBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(DenseDropoutBlock, self).__init__()
        self._linear = nn.Linear(in_features, out_features)
        self._no_linearity = NoLinearityBlock(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        return self._no_linearity(x)
