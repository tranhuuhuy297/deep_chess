import os
import torch
import gdown
import numpy as np
from torch.nn import functional as F


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



