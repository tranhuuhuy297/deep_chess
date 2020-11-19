import os
import torch
import numpy as np
from utils import download_weights
from model.auto_encoder import AE

games = []

for file in os.listdir('./data/bitboard'):
    games.append(file)

weight = download_weights('https://drive.google.com/uc?export=download&id=1oJvyy4Ec3vYmZOqN54F7_PrqY-DXdyvs')

model = AE()

state = torch.load(weight, map_location=lambda storage, loc: storage.cuda(0))

model.load_state_dict(state)

def featurize(game):
    _, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    return enc.detach().numpy()

for game in list(games):
    temp = np.load('data/bitboard/' + game)

    # len: 554116 for 4000 epoch parse data
    batched_games = np.split(temp, 19)

    feat_games = [featurize(batch) for batch in batched_games]
    featurized = np.vstack(feat_games)

    os.remove('data/bitboard/' + game)
    np.save('./data/features.npy', featurized)

