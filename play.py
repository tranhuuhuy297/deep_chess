import torch
import numpy as np
from model.auto_encoder import AE
from model.siamese import Siamese
from utils import download_weights, featurize, compare, gen_compare_array


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ae_weight = download_weights('https://drive.google.com/uc?export=download&id=1ZsRsjDF8T3e44JwpIkS5drWjtfqocDrv')
Siamese_weight = download_weights('https://drive.google.com/uc?export=download&id=1-MaSz_KX2xkYQJzL8f5-aA5kyamz_pjr')

ae_state = torch.load(ae_weight, map_location=lambda storage, loc: storage.cuda(0))
siamese_state = torch.load(Siamese_weight, map_location=lambda storage, loc: storage.cuda(0))