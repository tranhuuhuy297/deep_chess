import os
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset


class TrainSet(Dataset):
    def __init__(self, train_games):
        super().__init__()
        self.train_games = train_games

    def __getitem__(self, index):
        return torch.from_numpy(self.train_games[index]).type(torch.FloatTensor), 0

    def __len__(self):
        return self.train_games.shape[0]


class TestSet(Dataset):
    def __init__(self, test_games):
        super().__init__()
        self.test_games = test_games

    def __getitem__(self, index):
        return torch.from_numpy(self.test_games[index]).type(torch.FloatTensor), 0

    def __len__(self):
        return self.test_games.shape[0]
        

def loss_AE(pred, target):
    BCE = F.mse_loss(pred, target.view(-1, 773), size_average=False)
    return BCE

def loss_model(pred, target):
    BCE = F.binary_cross_entropy(pred, target.view(-1, 2), size_average=False)
    return BCE

def save_weight(model, optimizer, epoch, save_dir = 'checkpoints/auto_encoder/'):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, 'ae_{}.pth.tar'.format(epoch)))