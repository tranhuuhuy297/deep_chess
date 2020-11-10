import os
import torch
import argparse
import numpy as np
from torch import nn, optim
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from utils import loss_AE
from model.siamese import Siamese


parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=512,
                    help='batch_size for training, default=512')
parser.add_argument('--epoch', type=int, default=1000,
                    help='number epochs for training, default=200')
parser.add_argument('--lr', type=float, default=0.01,
                    help='lr begin from 0.005, *0.98 after epoch')
parser.add_argument('--decay', type=float, default=0.99,
                    help='decay rate of lr, default=0.98')
parser.add_argument('--seed', type=int, default=42,
                    help='seed number for random, default=42')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)

writer = SummaryWriter(comment='Lr: {} | batch_size: {}'.format(args.lr, args.batch))

print("Loading data...")
