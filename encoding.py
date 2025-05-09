import random
from typing import Type, Callable
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn, optim
import data as d
import torch.nn.functional as F

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class EncodingModel(nn.Module):

    def __init__(self, channels, kernels, fc, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = d.NUM_VALUES
        for channel, kernel in zip(channels, kernels):
            self.layers.append(nn.Conv2d(in_channels=in_size, out_channels=channel,
                                         kernel_size=kernel, padding="same"))
            in_size = channel
        self.pool = nn.MaxPool2d(d.MAX_X, d.MAX_Y)
        self.fc = nn.Linear(in_size, fc)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.dropout(x)
        #x = x.mul(mask)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

def encode(model: nn.Module, batch: tuple, device: str=DEVICE):
    (anchor, positive, negative, anchor_mask, comp_mask) = batch
    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)
    anchor_mask = anchor_mask.to(device)
    comp_mask = comp_mask.to(device)
    anchor_encoding = model(anchor, anchor_mask)
    positive_encoding = model(positive, comp_mask)
    negative_encoding = model(negative, comp_mask)
    return anchor_encoding, positive_encoding, negative_encoding

def train_epoch(model: nn.Module, data: d.ArcTriplets, loss_fn: nn.Module, optimizer: optim.Optimizer,
                batch_size: int=128, device: str=DEVICE, log: bool=True, collate_fn: Callable=None):
    model.train()
    size = len(data)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)
    for i, batch in enumerate(loader):
        anchor_encoding, positive_encoding, negative_encoding = encode(model, batch, device)
        loss = loss_fn(anchor_encoding, positive_encoding, negative_encoding)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if log and i % 10 == 0:
            loss = loss.item()
            current = (i + 1) * len(batch[0])
            print(f"loss: {loss:>7f}, [{current:>6d}/{size:>6d}]", end="\r")

def test(model: nn.Module, data: d.ArcTriplets, loss_fn: nn.Module, batch_size: int=128,
         device: str=DEVICE, collate_fn: Callable=None):
    model.eval()
    loss = 0
    size = len(data)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            anchor_encoding, positive_encoding, negative_encoding = encode(model, batch, device)
            loss += loss_fn(anchor_encoding, positive_encoding, negative_encoding) * len(batch[0])
    loss /= size
    return loss

def train(model: nn.Module, data: d.ArcTriplets, loss_fn: nn.Module, optimizer: optim.Optimizer,
          name: str="model", max_epochs: int=100, max_streak: int=5,
          seed: int=0, device: str=DEVICE, train_size: float=0.9,
          batch_size:int=128, log:bool=True):
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
        train_loss = test(model, train_data, loss_fn, batch_size, device)
        val_loss = test(model, val_data, loss_fn, batch_size, device)
        if val_loss < best_loss:
            improved = True
            best_loss = val_loss
        else:
            improved = False
        if log:
            print(f"Epoch {i:3d}: Loss={val_loss:>3.5f} val, {train_loss:>3.5f} train ")
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
