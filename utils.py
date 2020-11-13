import os
import torch
import gdown
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset


class TrainSet_AE(Dataset):
    def __init__(self, train_games):
        super().__init__()
        self.train_games = train_games

    def __getitem__(self, index):
        return torch.from_numpy(self.train_games[index]).type(torch.FloatTensor), 0

    def __len__(self):
        return self.train_games.shape[0]


class TestSet_AE(Dataset):
    def __init__(self, test_games):
        super().__init__()
        self.test_games = test_games

    def __getitem__(self, index):
        return torch.from_numpy(self.test_games[index]).type(torch.FloatTensor), 0

    def __len__(self):
        return self.test_games.shape[0]
        

class TrainSet(Dataset):
    def __init__(self, train_games_win, train_games_loss):
        self.train_games_win = train_games_win
        self.train_games_loss = train_games_loss

    def __getitem__(self, index):
        rand_win = self.train_games_win[
            np.random.randint(0, self.train_games_win.shape[0])]
        rand_loss = self.train_games_loss[
            np.random.randint(0, self.train_games_loss.shape[0])]

        order = np.random.randint(0,2)
        if order == 0:
            stacked = np.hstack((rand_win, rand_loss))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return stacked, label
        else:
            stacked = np.hstack((rand_loss, rand_win))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
            return stacked, label

    def __len__(self):
        return len(self.train_games_win) + len(self.train_games_loss)


class TestSet(Dataset):
    def __init__(self, test_games_win, test_games_loss):
        self.test_games_win = test_games_win
        self.test_games_loss = test_games_loss

    def __getitem__(self, index):
        rand_win = self.test_games_win[np.random.randint(0, self.test_games_win.shape[0])]
        rand_loss = self.test_games_loss[np.random.randint(0, self.test_games_loss.shape[0])]

        order = np.random.randint(0,2)
        if order == 0:
            stacked = np.hstack((rand_win, rand_loss))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return stacked, label
        else:
            stacked = np.hstack((rand_loss, rand_win))
            stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
            return stacked, label

    def __len__(self):
        return len(self.test_games_win) + len(self.test_games_loss)


def loss_AE(pred, target):
    BCE = F.mse_loss(pred, target.view(-1, 773), size_average=False)
    return BCE

def loss_model(pred, target):
    BCE = F.binary_cross_entropy(pred, target.view(-1, 2), size_average=False)
    return BCE

def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?export=download&id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)