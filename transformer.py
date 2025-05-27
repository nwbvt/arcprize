import data
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
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
        res = rearrange(res, 'b x y c -> b c x y')
        return res

def train_epoch(model: TransformerModel, data: d.ArcSequence, loss_fn: nn.Module, optimizer: optim.Optimizer,
                batch_size: int=4, device: str=DEVICE, log: bool=True):
    model.train()
    size = len(data)
    loader = DataLoader(data, batch_size=batch_size)
    for i, (x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if log:
            loss = loss.item()
            current = (i + 1) * len(x)
            print(f"loss: {loss:>7f}, [{current:>6d}/{size:>6d}]", end="\r")

def test(model: TransformerModel, data: d.ArcSequence, loss_fn: nn.Module, batch_size: int=4, device: str=DEVICE):
    model.eval()
    loss = 0
    size = len(data)
    loader = DataLoader(data, batch_size=batch_size)
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss += loss_fn(preds, y)
    loss /= size
    return loss

def train(model: TransformerModel, data: d.ArcSequence, loss_fn: nn.Module, optimizer: optim.Optimizer,
          name: str="transformer_model", max_epochs: int=100, max_streak: int=5,
          seed: int=0, device: str=DEVICE, train_size: float=0.9,
          batch_size:int=4, log:bool=True):
    fname=f"{name}.pth"
    generator = None
    if seed:
        generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    train_data, val_data = torch.utils.data.random_split(data, [train_size, 1-train_size], generator)
    best_loss = test(model, val_data, loss_fn, batch_size, device)
    streak = 0 # Number of times the train loss has not improved
    for i in range(max_epochs):
        train_epoch(model, train_data, loss_fn, optimizer, batch_size, device, log)
        val_loss = test(model, val_data, loss_fn, batch_size, device)
        if val_loss < best_loss:
            improved = True
            best_loss = val_loss
        else:
            improved = False
        if log:
            print(f"Epoch {i:3d}: Loss={val_loss:>3.5f}")
        if improved:
            streak = 0
            torch.save(model.state_dict(), fname)
        else:
            streak += 1
            if streak >= max_streak:
                break
    model.load_state_dict(torch.load(fname))
    final_loss = test(model, val_data, loss_fn, batch_size, device)
    print(f"Final Loss: {final_loss:>3.5f}")
    return model

