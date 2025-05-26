import data
import torch
from torch import nn, optim, Tensor
from einops import rearrange
import data as d
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding3D

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class TransformerModel(nn.Module):
    def __init__(self, embedder, d_model, n_head, n_hid, n_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedder = embedder
        d_embed = embedder.out_size + data.NUM_VALUES + 1
        self.positional_encoder = PositionalEncoding3D(d_embed)
        self.pre_encode = nn.Linear(d_embed, d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=n_hid, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, n_layers, nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(d_model, data.NUM_VALUES+1)

    def forward(self, x: Tensor) -> Tensor:
        batch, n_seq, n_vals, n_x, n_y = x.shape
        assert(n_vals == data.NUM_VALUES+1)
        assert(n_x == data.MAX_X)
        assert(n_y == data.MAX_Y)
        x = x.reshape((batch*n_seq, n_vals, n_x, n_y))
        res = self.dropout(torch.cat([x, self.embedder(x)], dim=1))
        res = res.reshape((batch, n_seq, self.embedder.out_size+data.NUM_VALUES+1, n_x, n_y))
        res = rearrange(res, 'b s c x y -> b s x y c')
        res = self.positional_encoder(res)
        res = rearrange(res, 'b s x y c -> b (s x y) c')
        res = self.pre_encode(res)
        res = self.transformer_encoder(res)
        res = rearrange(res, 'b (s x y) c -> b s x y c', x=data.MAX_X, y=data.MAX_Y)
        res = res[:,-1,:,:,:]
        res = self.decoder(res)
        res = F.log_softmax(res, dim=-1)
        return res

