import os
import torch
import argparse
import numpy as np
from torch import optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
from model.siamese import Siamese
from utils import loss_model
from data_generator import TrainSet, TestSet


parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=512,
                    help='batch_size for training, default=512')
parser.add_argument('--epoch', type=int, default=500,
                    help='number epochs for training, default=500')
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

games = []
for file in os.listdir('./data/bitboard'):
    games.append(file)

labels = []
for file in os.listdir('./data/label'):
    labels.append(file)

train_loader = data.DataLoader(TrainSet(games, labels), batch_size=args.batch, shuffle=True)
test_loader = data.DataLoader(TestSet(games, labels), batch_size=args.batch, shuffle=True)

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
    writer.add_scalar('test_loss', test_loss, epoch)
    
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, args.epoch + 1):
    train(epoch)
    test(epoch)

    # Adjust learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * args.decay

torch.save(model.state_dict(), 'siamese.pth')