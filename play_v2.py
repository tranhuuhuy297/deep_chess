import pygame as p

import ChessEngine

WIDTH = HEIGHT = 512
DEMENSION = 8  # 8X8
SQ_SIZE = HEIGHT // DEMENSION
MAX_FPS = 15
IMAGES = {}  # khai báo dict


def loadImages():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        # load and scale image
        IMAGES[piece] = p.transform.scale(p.image.load("app/static/images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))


def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    p.display.set_caption('Chess')

    # Game state
    gs = ChessEngine.GameState()
    minmaxPruning = ChessEngine.MinMaxPruning()

    validMoves = gs.getValidMoves()
    moveMade = False
    loadImages()

    # loadImage()
    running = True
    sqSelected = ()  # tuple (row, col)
    playerClicks = []
    while running:
        if gs.whiteToMove:
            for e in p.event.get():
                if e.type == p.QUIT:
                    running = False
                # mouse handler
                elif e.type == p.MOUSEBUTTONDOWN:
                    location = p.mouse.get_pos()  # (x, y) location of mouse
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if sqSelected == (row, col):  # player clicked the same square
                        sqSelected = ()  # deselect
                        playerClicks = []  # clear player click
                    else:
                        sqSelected = (row, col)
                        playerClicks.append(sqSelected)  # apend fot 1st and 2nd click
                    if len(playerClicks) == 2:  # after player's 2nd click
                        move = ChessEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                        if move in validMoves:
                            gs.makeMove(move)
                            moveMade = True
                            sqSelected = ()  # reset player clicks
                            playerClicks = []
                        else:
                            playerClicks = [sqSelected]
                # key handler
                elif e.type == p.KEYDOWN:
                    if e.key == p.K_z:
                        gs.undoMove()
                        moveMade = True
            if moveMade:
                validMoves = gs.getValidMoves()
                moveMade = False
        else:
            # print(minmaxPruning.minmaxRoot(2, gs))
            bestMove = minmaxPruning.minmaxRoot(3, gs)
            gs.makeMove(bestMove)
            print("best move: ", bestMove.getChessNotation())
            print(gs)
            moveMade = True
            sqSelected = ()  # reset player clicks
            playerClicks = []
            print("luot cua trang: ", gs.whiteToMove)

        # drawGameState(screen, gs)
        drawGameState(screen, gs, validMoves, sqSelected)

        clock.tick(MAX_FPS)
        p.display.flip()


'''
highlight selected position and move animation
'''


def highlightSquare(screen, gs, validMoves, sqSelected):
    if sqSelected != ():
        row, col = sqSelected
        if gs.board[row][col][0] == ('w' if gs.whiteToMove else 'b'):  # consider the color
            # highlight selected square
            surface = p.Surface((SQ_SIZE, SQ_SIZE))
            surface.set_alpha(100)  # transperancy value
            surface.fill(p.Color("blue"))
            screen.blit(surface, (col * SQ_SIZE, row * SQ_SIZE))
            # highlight posible moves
            surface.fill(p.Color("yellow"))
            for move in validMoves:
                if move.startRow == row and move.startCol == col:
                    screen.blit(surface, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))


def drawGameState(screen, gs, validMoves, sqSelected):
    drawBoard(screen)  # draw square on board

    drawPieces(screen, gs.board)  # draw pieces
    highlightSquare(screen, gs, validMoves, sqSelected)


'''
draw square on board
'''


def drawBoard(screen):
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DEMENSION):
        for c in range(DEMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


'''
draw pieces on board
'''


def drawPieces(screen, board):
    for r in range(DEMENSION):
        for c in range(DEMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


if __name__ == "__main__":
    main()