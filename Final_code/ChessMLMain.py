# ChessMLMain.py
import sys, pygame as p, random, torch
import ChessEngine, constraint as C
from MLAgent import load_model, encode_board

def loadImages():
    for piece in ['wp','bp','wr','br','wn','bn','wb','bb','wq','bq','wk','bk']:
        C.IMAGES[piece] = p.transform.scale(
            p.image.load('images/'+piece+'.png'),
            (C.SQ_SIZE, C.SQ_SIZE)
        )

def coords_from_uci(uci):
    files = ChessEngine.Move.filesToCols
    ranks = ChessEngine.Move.ranksToRows
    sr, sc = ranks[uci[1]], files[uci[0]]
    er, ec = ranks[uci[3]], files[uci[2]]
    return (sr,sc),(er,ec)

def drawBoard(screen):
    colors = [p.Color('white'), p.Color('gray')]
    for r in range(C.DIMENSION):
        for c in range(C.DIMENSION):
            color = colors[(r+c)%2]
            p.draw.rect(screen, color,
                        p.Rect(c*C.SQ_SIZE, r*C.SQ_SIZE, C.SQ_SIZE, C.SQ_SIZE))

def drawPieces(screen, board):
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece!='--':
                screen.blit(C.IMAGES[piece],
                            p.Rect(c*C.SQ_SIZE,r*C.SQ_SIZE,C.SQ_SIZE,C.SQ_SIZE))

def drawGameState(screen, gs):
    drawBoard(screen)
    drawPieces(screen, gs.board)

def drawText(screen, text):
    font = p.font.SysFont('Arial', 32, True, False)
    txt = font.render(text, True, p.Color('black'))
    loc = txt.get_rect(center=(C.WIDTH//2, C.HEIGHT//2))
    screen.blit(txt, loc)

def main(model_path):
    p.init()
    screen = p.display.set_mode((C.WIDTH, C.HEIGHT))
    clock = p.time.Clock()
    loadImages()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, move2idx, idx2move = load_model(model_path, device)

    gs = ChessEngine.GameState()
    gameOver = False

    while True:
        for e in p.event.get():
            if e.type == p.QUIT:
                p.quit()
                sys.exit()

        if not gameOver:
            ml_turn = gs.whiteToMove  # ML luôn cầm trắng
            valid = gs.getValidMoves()

            if valid:
                if ml_turn:
                    print("ML Agent's turn")
                    x = encode_board(gs.board).to(device)
                    with torch.no_grad():
                        logits = model(x).squeeze()
                    legal_ids = []
                    for m in valid:
                        u = m.getRankFile(m.startRow,m.startCol) + m.getRankFile(m.endRow,m.endCol)
                        if u in move2idx:
                            legal_ids.append(move2idx[u])
                    if legal_ids:
                        best = legal_ids[logits[legal_ids].argmax().item()]
                        u = idx2move[best]
                        (sr,sc),(er,ec) = coords_from_uci(u)
                        move = ChessEngine.Move((sr,sc),(er,ec), gs.board)
                        if move.isPromote:
                            move.promoteTo = 'q'
                    else:
                        move = random.choice(valid)
                else:
                    move = random.choice(valid)  # Random cầm đen

                gs.makeMove(move)
                p.time.delay(300)

            else:
                gameOver = True

        drawGameState(screen, gs)

        if gameOver:
            if gs.checkMate:
                winner = 'ML Agent (White)' if not gs.whiteToMove else 'Random (Black)'
                drawText(screen, f"{winner} wins!")
            else:
                drawText(screen, "Draw!")

        p.display.flip()
        clock.tick(C.MAX_FPS)

if __name__ == "__main__":
    main("policy_net.pth")