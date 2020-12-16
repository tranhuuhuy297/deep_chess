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

        for board in list(self.train_games):
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

        for board in list(self.test_games):
            temp = np.load('data/bitboard/' + board)
            self.bitboard = temp[int(len(temp)*.85):]
            np.random.shuffle(self.bitboard)

    def __getitem__(self, index):
        return torch.from_numpy(self.bitboard[index]).type(torch.FloatTensor), 0

    def __len__(self):
        return self.bitboard.shape[0]
        

# dataset for Siamese
class TrainSet(Dataset):
    def __init__(self, games, labels):
        super().__init__()
        self.games = games
        self.labels = labels
        self.train_games = None
        self.train_labels_win = None
        self.train_games_win = None
        self.train_games_loss = None

        for file in list(self.games):
            game = np.load(file)
            self.train_games = game[:int(len(game)*.85)]
        
        for file in list(self.labels):
            label = np.load(file)
            self.train_label_win = label[:int(len(label)*.85)]

        self.train_games_win = self.train_games[self.train_label_win == 1]
        self.train_games_loss = self.train_games[self.train_label_win == -1]

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
    def __init__(self, games, labels):
        super().__init__()
        self.games = games
        self.labels = labels
        self.test_games = None
        self.test_labels_win = None
        self.test_games_win = None
        self.test_games_loss = None

        for file in list(self.games):
            game = np.load(file)
            self.test_games = game[int(len(game)*.85):]
        
        for file in list(self.labels):
            label = np.load(file)
            self.test_label_win = label[int(len(label)*.85):]

        self.test_games_win = self.test_games[self.test_label_win == 1]
        self.test_games_loss = self.test_games[self.test_label_win == -1]

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