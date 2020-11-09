import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset


def loss_AE(pred, target):
    BCE = F.mse_loss(pred, target.view(-1, 773), size_average=False)
    return BCE

def loss_model(pred, target):
    BCE = F.binary_cross_entropy(pred, target.view(-1, 2), size_average=False)
    return BCE
