import data
import torch
from torch import nn, optim
import data as d
import torch.nn.functional as F

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class SolvingModel(nn.Module):
    def __init__(self, encoder, size, kernel, dropout):
        super().__init__()
        self.encoder = encoder
        self.conv = nn.Conv2d(encoder.out_size, size, kernel, padding="same")
        self.final = nn.Linear(size, data.NUM_VALUES)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        encoded = self.encoder(x)
        hidden = F.relu(self.dropout(self.conv(encoded)))
        y = self.final(hidden.permute([0,2,3,1]))
        return y.permute([0,3,1,2])

def train_mini_model(train_inputs, train_outputs, model, n_epochs=10, lr=0.001, l2=0.01):
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    train_inputs = train_inputs.to(DEVICE)
    train_outputs = train_outputs.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for i in range(n_epochs):
        preds = model(train_inputs)
        loss = criterion(preds.flatten(start_dim=2, end_dim=3), train_outputs.flatten(start_dim=2, end_dim=3))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        accuracy = (1-((preds.softmax(dim=1) > 0.5).to(torch.long) - train_outputs).abs().mean())*100
        print(f"loss: {loss:>7f}, accuracy: {accuracy:>5f}% [{i+1}/{n_epochs}]", end="\r")


