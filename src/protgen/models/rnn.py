from pathlib import Path

import torch
from torch.nn import init
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
# from sklearn.model_selection import GridSearchCV
# import numpy as np
# import sys

from protgen.data import ProteinDataset
from protgen.data.make_data import IGNORE_INDEX

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

class GRU(nn.Module):

    def __init__(self, vocab_size=23, embed_dim=8, hidden_size=64, num_layers=1):
        super(GRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.GRU = torch.nn.GRU(embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.activation_output = torch.nn.LogSoftmax(dim=-1)

    def forward(self, context):
        x = self.embed(context)
        hidden_states, final_hidden_state = self.GRU(x)
        logits = self.output(hidden_states)
        probs = self.activation_output(logits)
        return probs

def train_rnn( model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, x_shifted, _, _ in train_loader:
        x = x.to(device)
        x_shifted = x_shifted.to(device)
        optimizer.zero_grad()
        log_probs = model(x)
        predictions = log_probs.reshape(-1, log_probs.shape[-1])
        targets = x_shifted.reshape(-1)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_rnn(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, x_shifted, _, _ in val_loader:
            x = x.to(device)
            x_shifted = x_shifted.to(device)
            log_probs = model(x)
            predictions = log_probs.reshape(-1, log_probs.shape[-1])
            targets = x_shifted.reshape(-1)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    train_data = ProteinDataset(path_to_data=DATA_DIR / "train.csv")
    val_data = ProteinDataset(path_to_data=DATA_DIR / "val.csv")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    print(f"Training examples: {len(train_data):,}")
    print(f"Training batches per epoch: {len(train_loader):,}")
    # epoch_run = [5,10,20]
    # learning_rate_runs = [1e-1,1e-2,1e-3]
    # hidden_size_runs = [64,128]
    # hidden_layer_run = [1,2]
    # param_grid = {'epoch': epoch_run,
    #               'lr': learning_rate_runs,
    #               'hidden_size': hidden_size_runs,
    #               'hidden_layer': hidden_layer_run}
    device = torch.device("cpu")
    # print("modello non LEETTTOOOOOOOO")
    # model = GRU().to(device)
    # grid = GridSearchCV(estimator = model, param_grid= param_grid, n_jobs=-1, cv = 3)
    # n_epochs, lr, hidden_size, hidden_layer = grid.fit()
    # print("li ho letti")
    n_epochs, lr, hidden_size, num_layers = 10, 1e-3, 64, 1
    model = GRU(hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.NLLLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    RESULTS_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    best_path = RESULTS_DIR / "rnn_latest.pt"
    for epoch in range(n_epochs):
        loss = train_rnn(model, train_loader, optimizer, criterion, device)
        loss_val = evaluate_rnn(model, val_loader, criterion, device)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch + 1 }/{n_epochs} training loss: {loss:.4f}")
        print(f"Epoch {epoch + 1 }/{n_epochs} validation loss: {loss_val:.4f}")

    torch.save(model.state_dict(), RESULTS_DIR / "rnn.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    best_loss = evaluate_rnn(model, val_loader, criterion, device)
    print(f"Reloaded checkpoint validation loss: {best_loss:.4f}")
