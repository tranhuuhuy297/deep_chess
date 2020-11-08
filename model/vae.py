import torch
import argparse
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision import datasets, transforms


parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=512,
                    help='batch size for training, default=512')
parser.add_argument('--epoch', type=int, default=100,
                    help='number epochs to train, default=100')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='enable gpu to train')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for split data')

args = parser.parse_args()

if(torch.cuda.is_available()==False): args.gpu = False

torch.manual_seed(args.seed)

device = torch.device('cuda' if args.gpu else 'cpu')

writer = SummaryWriter()

games = np.load()