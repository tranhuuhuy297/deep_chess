import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset


games = np.load('../data/bilboards.npy')
np.random.shuffle(games)

train = games[:int(len(games)*0.85)]
test = games[int(len(games)*0.85):]

class TrainSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return torch.from_numpy(train[index].type(torch.FloatTensor), 1)

    def __len__(self):
        return train.shape[0]


def loss_AE(pred, target):
    BCE = F.binary_cross_entropy(pred, target.view(-1, 773), size_average=False)
    return BCE

def loss_model(pred, target):
    BCE = F.binary_cross_entropy(pred, target.view(-1, 2), size_average=False)
    return BCE
