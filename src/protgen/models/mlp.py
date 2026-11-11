import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

from protgen.data.make_data import IGNORE_INDEX 

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[3] / "data"


train_data = ProteinDataset(path_to_data=DATA_DIR / "train.csv")
val_data = ProteinDataset(path_to_data=DATA_DIR / "val.csv")
test_data = ProteinDataset(path_to_data=DATA_DIR / "test.csv")

print(f"Data class: {train_data}")

# Get tensors
X_train = train_data.tokens
y_train_shifted = train_data.tokens_shifted # generative task

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # ignore padding index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

n_epochs = 10


class MLP(nn.Module):

    def __init__(self, vocab_size=23, context_size=3, embed_dim=32, num_hidden=256):
        super(MLP, self).__init__()  
        self.context_size = context_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        num_features = context_size * embed_dim
        # input layer L1 + hidden layer H1
        self.L1 = nn.Linear(num_features, num_hidden)
        self.H1 = nn.Linear(num_hidden, num_hidden)
        # output layer O1
        self.O1 = nn.Linear(num_hidden, vocab_size)
        self.activation = torch.nn.ELU()

    def forward(self, context):
        x = self.embed(context).flatten(1)
        x = self.L1(x)
        x = self.activation(x)
        x = self.H1(x)
        x = self.activation(x)
        x = self.O1(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_mlp(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, x_shifted, _, _ in train_loader:
        context, target = make_windows(x, x_shifted, model.context_size)
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_mlp(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, x_shifted, _, _ in test_loader:
            context, target = make_windows(x, x_shifted, model.context_size)
            context, target = context.to(device), target.to(device)
            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def make_windows(x, x_shifted, k):
    xp = F.pad(x, (k-1, 0), value=0)  
    context = xp.unfold(1, k, 1)
    target = x_shifted 
    context = context.reshape(-1, k)
    target = target.reshape(-1)
    return context, target

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


if __name__ == "__main__":
    train_data = ProteinDataset(path_to_data=DATA_DIR / "train.csv")
    val_data = ProteinDataset(path_to_data=DATA_DIR / "val.csv")
    test_data = ProteinDataset(path_to_data=DATA_DIR / "test.csv")

    print(f"Data class: {train_data}")

    # Get tensors
    X_train = train_data.tokens
    y_train_shifted = train_data.tokens_shifted # generative task

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # ignore padding index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    n_epochs = 10

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        train_loss = train_mlp(model, train_loader, optimizer, criterion, device)
        val_loss = test_mlp(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), DATA_DIR.parent / "results" / "mlp.pt")

