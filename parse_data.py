import os
import chess.pgn
import numpy as np


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

def get_result(game):
    result = game.headers['Result']
    result = result.split('-')
    if result[0] == '1':
        return 1
    elif result[0] == '0':
        return -1
    else:
        return 0

games = open('./data/CCRL-4040.[1187224].pgn')

if not os.path.isdir('./data/bitboard'):
    os.mkdir('./data/bitboard')

if not os.path.isdir('./data/label'):
    os.mkdir('./data/label')

num_games = 0
while (num_games<40000):
  bitboards = []
  labels = []
  for i in range(4000):
    num_games += 1
    game = chess.pgn.read_game(games)
    result = get_result(game)
    board = game.board()

    for move in game.main_line():
    # for move in game.mainline_moves():
        board.push(move)
        bitboard = get_bitboard(board)
        bitboards.append(bitboard)
        labels.append(result) 
  
  bitboards = np.array(bitboards)
  labels = np.array(labels)

  np.save('./data/bitboard/bitboards_' + str(num_games) + '.npy', bitboards)
  np.save('./data/label/labels_' + str(num_games) + '.npy', labels)

print('Done parse!')
print('Done save data to numpy file!')