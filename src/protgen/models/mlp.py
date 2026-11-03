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
        # input layer
        self.W_1 = Parameter(init.xavier_normal_(torch.empty(num_hidden, num_features)))
        self.b_1 = Parameter(torch.zeros(num_hidden))
        # hidden layer
        self.W_2 = Parameter(init.xavier_normal_(torch.empty(vocab_size, num_hidden)))
        self.b_2 = Parameter(torch.zeros(vocab_size))
        # define activation function in constructor
        self.activation = torch.nn.ELU()

    def forward(self, context):
        x = self.embed(context).flatten(1)
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = F.linear(x, self.W_2, self.b_2)
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
