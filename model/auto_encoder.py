import torch
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.fce1 = nn.Linear(773, 600)
        self.bne1 = nn.BatchNorm1d(600)
        self.fce2 = nn.Linear(600, 400)
        self.bne2 = nn.BatchNorm1d(400)
        self.fce3 = nn.Linear(400, 200)
        self.bne3 = nn.BatchNorm1d(200)
        self.fce4 = nn.Linear(200, 100)
        self.bne4 = nn.BatchNorm1d(100)

        # Decoder
        self.fcd1 = nn.Linear(100, 200)
        self.bnd1 = nn.BatchNorm1d(200)
        self.fcd2 = nn.Linear(200, 400)
        self.bnd2 = nn.BatchNorm1d(400)
        self.fcd3 = nn.Linear(400, 600)
        self.bnd3 = nn.BatchNorm1d(600)
        self.fcd4 = nn.Linear(600, 773)
        self.bnd4 = nn.BatchNorm1d(773)

    def encoder(self, x):
        x_e = F.leaky_relu(self.bne1(self.fce1(x)))
        x_e = F.leaky_relu(self.bne2(self.fce2(x_e)))
        x_e = F.leaky_relu(self.bne3(self.fce3(x_e)))
        x_e = F.leaky_relu(self.bne4(self.fce4(x_e)))

    def decoder(self, x_e):
        x_d = F.leaky_relu(self.bnd1(self.fcd1(x_e)))
        x_d = F.leaky_relu(self.bnd2(self.fcd2(x_d)))
        x_d = F.leaky_relu(self.bnd3(self.fcd3(x_d)))
        x_d = F.sigmoid(self.bnd4(self.fcd4(x_d)))
    
    def forward(self, x):
        input = self.encoder(x.view(-1, 773))
        output = self.decoder(input)
        return output, input

