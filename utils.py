import os
import torch
import gdown
import base64
import chess
import numpy as np
from torch.nn import functional as F


# 2 loss function 
def loss_AE(pred, target):
    BCE = F.mse_loss(pred, target.view(-1, 773), size_average=False)
    return BCE

def loss_model(pred, target):
    BCE = F.binary_cross_entropy(pred, target.view(-1, 2), size_average=False)
    return BCE

# download weight from my driver
def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?export=download&id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)

# Decode movement
def featurize(featurizer, boards, device):
    boards = torch.from_numpy(boards).type(torch.FloatTensor).to(device)
    return featurizer(boards)

# Compare which one is better for black to win player
def compare(comparator, features, device):
    features = torch.from_numpy(features).type(torch.FloatTensor).to(device)
    return comparator(features).cpu().detach().numpy()

# Gen all pair movements
def gen_compare_array(features):
    catcats = []
    for feature in features:
        cats = []
        for compared in features:
            cats.append(np.hstack((feature, compared)))
        cats = np.array(cats)
        catcats.append(cats)
    return np.vstack(catcats)

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

def convert(board):
    chess_board =  [["--", "--", "--", "--", "--", "--", "--", "--"],
                    ["--", "--", "--", "--", "--", "--", "--", "--"],
                    ["--", "--", "--", "--", "--", "--", "--", "--"],
                    ["--", "--", "--", "--", "--", "--", "--", "--"],
                    ["--", "--", "--", "--", "--", "--", "--", "--"],
                    ["--", "--", "--", "--", "--", "--", "--", "--"],
                    ["--", "--", "--", "--", "--", "--", "--", "--"],
                    ["--", "--", "--", "--", "--", "--", "--", "--"]]

    for i in range(64):
        row = int(i/8)
        col = i % 8
        if (board.piece_at(i) == None): chess_board[row][col] = "--"
        else : 
            if   (board.piece_at(i).symbol() == "R"): chess_board[row][col] = "wR"
            elif (board.piece_at(i).symbol() == "N"): chess_board[row][col] = "wN"
            elif (board.piece_at(i).symbol() == "B"): chess_board[row][col] = "wB"
            elif (board.piece_at(i).symbol() == "Q"): chess_board[row][col] = "wQ"
            elif (board.piece_at(i).symbol() == "K"): chess_board[row][col] = "wK"
            elif (board.piece_at(i).symbol() == "P"): chess_board[row][col] = "wp"

            elif (board.piece_at(i).symbol() == "r"): chess_board[row][col] = "bR"
            elif (board.piece_at(i).symbol() == "n"): chess_board[row][col] = "bN"
            elif (board.piece_at(i).symbol() == "b"): chess_board[row][col] = "bB"
            elif (board.piece_at(i).symbol() == "q"): chess_board[row][col] = "bQ"
            elif (board.piece_at(i).symbol() == "k"): chess_board[row][col] = "bK"
            elif (board.piece_at(i).symbol() == "p"): chess_board[row][col] = "bp"

    for i in range(int(len(chess_board)/2)):
        temp = chess_board[i]
        chess_board[i] = chess_board[len(chess_board)-1-i]
        chess_board[len(chess_board)-1-i] = temp

    return chess_board
