import torch
import numpy as np
from utils import download_weights
from model.auto_encoder import AE


weight = download_weights('https://drive.google.com/uc?export=download&id=1ZsRsjDF8T3e44JwpIkS5drWjtfqocDrv')

model = AE()

state = torch.load(weight, map_location=lambda storage, loc: storage.cuda(0))

model.load_state_dict(state)

games = np.load('./data/bitboards.npy')

# len: 554116 for 4000 epoch parse data
batched_games = np.split(games, 19)

def featurize(game):
    _, enc = model(torch.from_numpy(game).type(torch.FloatTensor))
    return enc.detach().numpy()

feat_games = [featurize(batch) for batch in batched_games]
featurized = np.vstack(feat_games)

np.save('./data/features.npy', featurized)