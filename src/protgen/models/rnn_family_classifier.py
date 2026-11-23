from pathlib import Path

import torch
from torch.nn import init
import torch.nn.functional as F
from torch import device, nn
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.data.make_data import IGNORE_INDEX
from protgen.models.rnn import GRU


class RNNClassifier(nn.Module):
    def __init__(self, pretrained_rnn: GRU, num_families = 3):
        super(RNNClassifier, self).__init__()
        self.embed = pretrained_rnn.embed
        self.embed.requires_grad_(False) 
        self.pretrained_rnn = pretrained_rnn.GRU
        self.pretrained_rnn.requires_grad_(False) # freeze weights of the pretrained RNN
        self.family_clasification_head = nn.Linear(self.pretrained_rnn.hidden_size, num_families) 
        self.classification_head_output = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x=self.embed(x)
        hidden_states, final_hidden_state = self.pretrained_rnn(x)
        logits = self.family_clasification_head(final_hidden_state[-1])
        probs = self.classification_head_output(logits)
        return probs

def train_classifier( model, train_loader, optimizer, device):
    criterion = nn.NLLLoss()
    model.train()
    total_loss, correct, n = 0, 0, 0
    for x, _, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)
        log_probs = model(x)
        targets = y.reshape(-1)
        loss = criterion(log_probs, targets)
        optimizer.zero_grad()
        loss.backward() # compute gradients
        optimizer.step() # take an optimization step
        total_loss += loss.item()
        correct += (log_probs.argmax(-1) == y).sum().item()
        n += len(y)
    return total_loss / len(train_loader), correct / n


def evaluate_classifier(model, val_loader, device):
    criterion = nn.NLLLoss()
    model.eval()
    total_loss, correct, n = 0, 0, 0
    with torch.no_grad():
        for x, _, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            log_probs = model(x)
            targets = y.reshape(-1)
            loss = criterion(log_probs, targets)
            total_loss += loss.item()
            correct += (log_probs.argmax(-1) == y).sum().item()
            n += len(y)
    return total_loss / len(val_loader), correct / n

if __name__ == "__main__":

    rnn = GRU()

    ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results"

    rnn.load_state_dict(torch.load(RESULTS_DIR / "rnn_latest.pt", map_location=device("cpu"), weights_only=True))

    train_data = ProteinDataset(path_to_data=DATA_DIR / "train.csv")
    val_data = ProteinDataset(path_to_data=DATA_DIR / "val.csv")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    print(f"Training examples: {len(train_data):,}")
    print(f"Training batches per epoch: {len(train_loader):,}")
    device = torch.device("cpu")
    model = RNNClassifier(rnn).to(device)
    criterion = nn.NLLLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.family_clasification_head.parameters(), lr=1e-3)
    n_epochs = 10
    RESULTS_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    best_path = RESULTS_DIR / "rnn_classifier_best.pt"
    for epoch in range(n_epochs):
        loss, acc = train_classifier(model, train_loader, optimizer, device)
        loss_val, acc_val = evaluate_classifier(model, val_loader, device)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch + 1 }/{n_epochs} training loss: {loss:.4f} accuracy: {acc:.4f}")
        print(f"Epoch {epoch + 1 }/{n_epochs} validation loss: {loss_val:.4f} accuracy: {acc_val:.4f}")

    torch.save(model.state_dict(), RESULTS_DIR / "rnn_classifier.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    best_loss, best_acc = evaluate_classifier(model, val_loader, device)
    print(f"Reloaded checkpoint validation loss: {best_loss:.4f} accuracy: {best_acc:.4f}")