import os
import torch
import argparse
import numpy as np
from torch import nn, optim
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from model.auto_encoder import AE
from utils import loss_AE, TrainSet_AE, TestSet_AE


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

train_loader = data.DataLoader(TrainSet_AE(train_games), batch_size=args.batch, shuffle=True)
test_loader  = data.DataLoader(TestSet_AE(test_games), batch_size=args.batch, shuffle=True)

model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    train_loss = 0.
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        dec, enc = model(data)
        loss = loss_AE(dec, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            writer.add_scalar('train_loss', loss.item() / len(data), epoch*len(train_loader) + batch_idx)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            dec, enc = model(data)
            test_loss += loss_AE(dec, data).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('test_loss', test_loss, epoch)


for epoch in range(1, args.epoch + 1):
    train(epoch)
    test(epoch)

    # Adjust learning rate
    for params in optimizer.param_groups:
        params['lr'] *= args.decay

# torch.save(model.state_dict(), 'ae.pth')