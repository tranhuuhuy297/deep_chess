import os
import torch
import argparse
import numpy as np
from torch import nn, optim
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from utils import loss_model, TrainSet, TestSet, get_acc
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

games = np.load('./data/features.npy')
labels = np.load('./data/labels.npy')

shuffle = np.random.permutation(len(labels))
games = games[shuffle]
labels = labels[shuffle]

train_games = games[:int(len(games)*.85)]
train_label_win = labels[: int(len(games)*.85)]
test_games = games[int(len(games)*.85):]
test_label_win = games[int(len(games)*.85):]

train_games_win = train_games[train_label_win == 1]
train_games_loss = train_games[train_label_win == -1]

test_games_win = test_games[test_label_win == 1]
test_games_loss = test_games[test_label_win == -1]

train_loader = torch.utils.data.DataLoader(TrainSet(train_games_win, train_games_loss),batch_size=args.batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestSet(test_games_win, test_games_loss),batch_size=args.batch, shuffle=True)

print('Building model...')
model = Siamese().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    train_loss = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_model(pred, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            writer.add_scalar('train_loss', loss.item() / len(data), epoch*len(train_loader) + batch_idx)
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

get_acc(model, device, test_loader)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            test_loss += loss_model(pred, label).item()

    test_loss /= len(test_loader.dataset)
    writer.add_scalar('data/test_loss', test_loss, epoch)
    
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, args.epoch + 1):
    train(epoch)
    test(epoch)

    # Adjust learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * args.decay

# torch.save(model.state_dict(), 'siamese.pth')