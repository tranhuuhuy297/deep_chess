import os
import time
import torch
import chess
import base64
import chess.svg
import traceback
import numpy as np
from flask import Flask, Response, request
from app import app
from app.state import State
from model.auto_encoder import AE
from model.siamese import Siamese
from utils import download_weights, featurize, compare, gen_compare_array


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ae_weight = download_weights('https://drive.google.com/uc?export=download&id=1ZsRsjDF8T3e44JwpIkS5drWjtfqocDrv')
siamese_weight = download_weights('https://drive.google.com/uc?export=download&id=1e76zvAoIz0d9xt3LCl5k0cwcpNwhEaJb')

ae_state = torch.load(ae_weight, map_location=lambda storage, loc: storage.cuda(0))
siamese_state = torch.load(siamese_weight, map_location=lambda storage, loc: storage.cuda(0))

featurizer = AE().to(device)
comparator = Siamese().to(device)

featurizer.eval()
comparator.eval()

s = State()

def get_bitboard(board):
    '''
    params
    ------
    board : chess.pgn board object
        board to get state from
    returns
    -------
    bitboard representation of the state of the game
    64 * 6 + 5 dim binary numpy vector
    64 squares, 6 pieces, '1' indicates the piece is at a square
    5 extra dimensions for castling rights queenside/kingside and whose turn
    '''
    bitboard = np.zeros(64*6*2 + 5)

    '''
    p: pawn
    n: knight
    b: bishop
    r: rook
    q: queen
    k: king
    '''
    piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

    for i in range(64):
        if board.piece_at(i):
            # White: return True
            # Black: return False
            color = int(board.piece_at(i).color) + 1
            bitboard[(piece_idx[board.piece_at(i).symbol().lower()] + i * 6) * color] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(chess.WHITE))
    bitboard[-3] = int(board.has_kingside_castling_rights(chess.BLACK))
    bitboard[-4] = int(board.has_queenside_castling_rights(chess.WHITE))
    bitboard[-5] = int(board.has_queenside_castling_rights(chess.BLACK))

    return bitboard

def get_best_move(board):
  moves = board.generate_legal_moves()
  moves = list(moves)

  bitboards = []
  for move in moves:
      b = board.copy()
      b.push(move)
      bitboards.append(get_bitboard(b))
  bitboards = np.array(bitboards)
  curr_bitboard = get_bitboard(board)

  _, features = featurize(featurizer, bitboards, device)
  features = features.cpu().detach().numpy()
  _, curr_features = featurize(featurizer, curr_bitboard, device)
  curr_features = curr_features.cpu().detach().numpy()

  to_compare = np.hstack((np.repeat(curr_features, len(moves), axis=0), features))
  scores = compare(comparator, to_compare, device)
  scores = scores[:, 1]
  best_idx = np.argmax(scores)
  board.push(moves[best_idx])

@app.route("/")
def hello():
  ret = open("./app/index.html").read()
  return ret.replace('start', s.board.fen())

# move given in algebraic notation
@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get('move',default="")
    if move is not None and move != "":
      print("Human moves:", move)
      try:
        s.board.push_san(move)
        get_best_move(s.board)
      except Exception:
        traceback.print_exc()
      response = app.response_class(
        response=s.board.fen(),
        status=200
      )
      return response
  else:
    print("GAME IS OVER")
    response = app.response_class(
      response="game over",
      status=200
    )
    return response
  return hello()

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print("Human moves:", move)
      try:
        s.board.push_san(move)
        get_best_move(s.board)
      except Exception:
        traceback.print_exc()
    response = app.response_class(
      response=s.board.fen(),
      status=200
    )
    return response

  print("GAME IS OVER")
  response = app.response_class(
    response="game over",
    status=200
  )
  return response

@app.route("/newgame")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response