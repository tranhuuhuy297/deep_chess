import torch
import numpy as np
from model.auto_encoder import AE

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = AE().to(device)

state = torch.load('./checkpoints/auto_encoder/ae_200.pth.tar', map_location=lambda storage, loc: storage.cuda(0))

model.load_state_dict(state['state_dict'])

games = np.load('./data/bitboards.npy')

# len: 554116 for 4000 epoch parse data
batched_games = np.split(games, 19)

def featurize(game):
    _, enc = model(torch.from_numpy(game).type(torch.FloatTensor).to(device))
    return enc.detach().numpy()

feat_games = [featurize(batch) for batch in batched_games]
featurized = np.vstack(feat_games)

np.save('./data/features.npy', featurized)