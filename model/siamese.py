import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        self.fc1 = nn.Linear(200, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 2)
        self.bn4 = nn.BatchNorm1d(2)

    def forward(self, x):
        output = F.leaky_relu(self.bn1(self.fc1(x)))
        output = F.leaky_relu(self.bn2(self.fc2(output)))
        output = F.leaky_relu(self.bn3(self.fc3(output)))
        output = F.sigmoid(self.bn4(self.fc4(output)))
        return output

    
