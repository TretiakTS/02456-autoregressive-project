import torch
from torch import device, nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

optimizer = torch.optim.Adam
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    for context, target in train_loader:
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
