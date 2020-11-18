import os
import torch
import gdown
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset


# Dataset for Auto encoder
class TrainSet_AE(Dataset):
    def __init__(self, train_games):
        super().__init__()
        self.train_games = train_games
        self.bitboard = None

        for board in list(train_games):
            temp = np.load('data/bitboard/' + board)
            self.bitboard = temp[:int(len(temp)*.85)]
            np.random.shuffle(self.bitboard)

    def __getitem__(self, index):
        return torch.from_numpy(self.bitboard[index]).type(torch.FloatTensor), 0

    def __len__(self):
        return self.bitboard.shape[0]


class TestSet_AE(Dataset):
    def __init__(self, test_games):
        super().__init__()
        self.test_games = test_games
        self.bitboard = None

        for board in list(test_games):
            temp = np.load('data/bitboard/' + board)
            self.bitboard = temp[int(len(temp)*.85):]
            np.random.shuffle(self.bitboard)

    def __getitem__(self, index):
        return torch.from_numpy(self.bitboard[index]).type(torch.FloatTensor), 0

    def __len__(self):
        return self.bitboard.shape[0]
        
# dataset for Siamese
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

# 2 loss function 
def loss_AE(pred, target):
    BCE = F.mse_loss(pred, target.view(-1, 773), size_average=False)
    return BCE

def loss_model(pred, target):
    BCE = F.binary_cross_entropy(pred, target.view(-1, 2), size_average=False)
    return BCE

# download weight from my driver
def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?export=download&id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)

# Decode movement
def featurize(featurizer, boards, device):
    boards = torch.from_numpy(boards).type(torch.FloatTensor).to(device)
    return featurizer(boards)

# Compare which one is better for black to win player
def compare(comparator, features, device):
    features = torch.from_numpy(features).type(torch.FloatTensor).to(device)
    return comparator(features).cpu().detach().numpy()

# Gen all pair movements
def gen_compare_array(features):
    catcats = []
    for feature in features:
        cats = []
        for compared in features:
            cats.append(np.hstack((feature, compared)))
        cats = np.array(cats)
        catcats.append(cats)
    return np.vstack(catcats)



