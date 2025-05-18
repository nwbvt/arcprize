import data
import torch
from torch import nn, optim
import data as d
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class TransformerModel(nn.Module):
    def __init__(self, encoder):
        self.encoder = encoder
        self.positional_encoder = PositionalEncoding2D(encoder.out_size + data.NUM_VALUES + 1)

    def forward(self, x: Tensor): -> Tensor:
        res = torch.cat([x, self.encoder(x)], dim=1)
        res = self.positional_encoder(res)

