import os
import torch
import argparse
import numpy as np
from torch import nn, optim
import torch.utils.data as data
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from utils import loss_AE
from model.auto_encoder import AE


parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=512,
                    help='batch_size for training, default=512')
parser.add_argument('--epoch', type=int, default=200,
                    help='number epochs for training, default=200')
parser.add_argument('--lr', type=float, default=0.005,
                    help='lr begin from 0.005, *0.98 after epoch')
parser.add_argument('--decay', type=float, default=0.98,
                    help='decay rate of lr, default=0.98')
parser.add_argument('--seed', type=int, default=42,
                    help='seed number for random, default=42')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)

writer = SummaryWriter(comment='Lr: {} | batch_size: {}'.format(args.lr, args.batch))

print("Loading data...")
games = np.load('./data/bitboards.npy')

np.random.shuffle(games)

train_games = games[:int(len(games)*.85)]
test_games  = games[int(len(games)*.85):]

class TrainSet(Dataset):
    def __init__(self):
        super().__init__
        pass

    def __getitem__(self, index):
        return torch.from_numpy(train_games[index]).type(torch.FloatTensor)

    def __len__(self):
        return train_games.shape[0]

class TestSet(Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __getitem__(self, index):
        return torch.from_numpy(test_games[index]).type(torch.FloatTensor)

    def __len__(self):
        return test_games.shape[0]

train = data.DataLoader(TrainSet(), batch_size=args.batch, shuffle=True)
test  = data.DataLoader(TestSet(), batch_size=args.batch, shuffle=True)

model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    train_loss = 0.
    for batch_idx, (data, _) in enumerate(train):
        data = data.to(device)
        optimizer.zero_grad()
        dec, enc = model(data)
        loss = loss_AE(dec, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train.dataset),
                100. * batch_idx / len(train),
                loss.item() / len(data)))
            writer.add_scalar('./data/train_loss', loss.item() / len(data), epoch*len(train) + batch_idx)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train.dataset)))


def save(epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch + 1}
    save_dir = './checkpoints/autoencoder/lr_{}_decay_{}'.format(int(args.lr*1000), int(args.decay*100))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, os.path.join(save_dir, 'ae_{}.pth.tar'.format(epoch)))


def dec(game):
    dec, _ = model(torch.from_numpy(game).type(torch.FloatTensor))
    dec = (dec.cpu().detach().numpy() > .5).astype(int)
    return dec


for epoch in range(1, args.epoch + 1):
    train(epoch)
    save(epoch)

    # Adjust learning rate
    for params in optimizer.param_groups:
        params['lr'] *= args.decay