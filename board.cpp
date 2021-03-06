#include "board.h"
#include "stdio.h"
#include "stdlib.h"

Board::Board () {
    hashes = (unsigned char*)malloc(sizeof(unsigned char));
    reset();
}

void Board::reset () {
    turn_ = WHITE;
    free(hashes);
    state_.clear();
    state_ = { {WR, WP, EM, EM, EM, EM, BP, BR},
               {WN, WP, EM, EM, EM, EM, BP, BN},
               {WB, WP, EM, EM, EM, EM, BP, BB},
               {WQ, WP, EM, EM, EM, EM, BP, BQ},
               {WK, WP, EM, EM, EM, EM, BP, BK},
               {WB, WP, EM, EM, EM, EM, BP, BB},
               {WN, WP, EM, EM, EM, EM, BP, BN},
               {WR, WP, EM, EM, EM, EM, BP, BR}
             };
    moves_.clear();
    states_.clear();
    hashes = (unsigned char*)malloc(8192 * sizeof(unsigned char));
    lastHash = 0;
    for (int i = 0; i < 8192; i++) {
        hashes[i] = 0;
    }
}

Board::Piece Board::getPiece (Board::Pos pos) {
    return state_ [pos.x - 1][pos.y - 1];
}

int Board::doMove (Board::Move move) {
    

    if (move.piece == EM) {
        fprintf(stderr, "Invalid move: An empty position was selected to move\n");
        return -1;
    }
    if (move.startPos.x < 1 || move.startPos.x > 8 || move.startPos.y < 1 || move.startPos.y > 8) {
        fprintf(stderr, "Invalid move: Selected start position is outside the range of the board [x, y] : {1, 8}\n");
        return -1;
    }
    if (getPiece(move.startPos) != move.piece) {
        fprintf(stderr, "Invalid move: The piece in the starting position is not the piece selected\n");
        return -1;
    }
    if (turn_ == WHITE && (move.piece <= BP) && (move.piece >= BK)) {
        fprintf(stderr, "Invalid move: It is white's turn to move, but a black piece was selected\n");
        return -1;
    }
    if (turn_ == BLACK && (move.piece <= WP) && (move.piece >= WK)) {
        fprintf(stderr, "Invalid move: It is black's turn to move, but a white piece was selected\n");
        return -1;
    }
    

    // player has selected a valid piece. Check to see if the place they're moving it to is valid.
    
    if (move.endPos.x < 1 || move.endPos.x > 8 || move.endPos.y < 1 || move.endPos.y > 8) {
        fprintf(stderr, "Invalid move: Selected end position is outside the range of the board [x, y] : {1, 8}\n");
        return -1;
    }

    if (move.piece != WP && move.piece != BP) {
        if (getPiece(move.endPos) != move.taken) {
            fprintf(stderr, "Invalid move: The piece in the ending position is not the piece specified\n");
            return -1;
        }
    }
    if (!Board::isLegalMove(move, true)) {
        return -1;
    }


    moves_.push_back(move);
    states_.push_back(state_);

    if (move.taken == EM && (move.startPos.x - move.endPos.x) != 0 && (move.piece == BP || move.piece == WP)) {
        // have already verified this is a valid move, there must have been
        // an en passant.
        Board::Pos pos = {.x = move.endPos.x, .y = move.startPos.y};
        Board::setPiece(pos, EM);
    }
    if ((move.piece == BK || move.piece == WK) && abs(move.startPos.x - move.endPos.x) == 2) {
        // have already verified this is a valid move, there must have been
        // a castle
        Board::Pos tmp = move.endPos;
        if ((move.endPos.x - move.startPos.x) > 0) {
            tmp.x += 1;
            Board::setPiece(tmp, EM);
            tmp.x -= 2;
            Board::setPiece(tmp, move.piece == BK ? BR : WR);
        }
        if ((move.endPos.x - move.startPos.x) < 0) {
            tmp.x -= 2;
            Board::setPiece(tmp, EM);
            tmp.x += 3;
            Board::setPiece(tmp, move.piece == BK ? BR : WR);
        }
    }
    Board::setPiece(move.startPos, EM);
    Board::setPiece(move.endPos, move.piece);
    // if the moved piece was a pawn that reached the end of the board
    // upgrade it to a queen    
    if (move.piece == WP && move.endPos.y == 8) {
        Board::setPiece(move.endPos, WQ);
    } else if (move.piece == BP && move.endPos.y == 1) {
        Board::setPiece(move.endPos, BQ);
    }



    Board::switchPlayer();
    Board::isCheck(turn_, true);   // get console log of check

    Board::hashBoard();
    unsigned char occurances = hashes[lastHash];

    if (Board::isCheckmate(turn_)){
        // last player to move just won the game
        printf("Checkmate, %s wins\n", turn_ == WHITE ? "black" : "white");
        if (turn_ == WHITE) {
            // black just won
            return 2;
        }
        if (turn_ == BLACK) {
            // white just won
            return 1;
        }
    }

    if (Board::isDraw(occurances)) {
        return -1;
    }

    // valid move, game is still continuing
    return 0;
}

// checks several scenarios where the game is unwinnable. see https://en.wikipedia.org/wiki/Draw_(chess)
// for more information
bool Board::isDraw (unsigned char occurances) {
    std::vector <Board::Move> moves;
    getLegalMoves(turn_, &moves);
    if (!Board::isCheck(turn_, false) && moves.size() == 0) {
        // player is not in check but has no legal moves. Stalemate
        return true;
    }


    if (occurances >= 3) {
        // same board has repeated 3 times
        return true;
    }

    int pC[13] = {0};

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            pC[state_[i][j]]++;
        }
    }
    if (pC[WQ] == 0 && pC[BQ] == 0 && pC[BP] == 0 && pC[WP] == 0) {
        // no pawns or queens
        if (pC[WB] == 0 && pC[BB] == 0 && pC[BN] == 0 && pC[WN] == 0 && pC[BR] == 0 && pC[WR] == 0) {
            // only kings remain
            return true;
        }
        if ((pC[WB] == 1 && pC[BB] == 0 || pC[WB] == 1 && pC[BB] == 0) && pC[BN] == 0 && pC[WN] == 0 && pC[BR] == 0 && pC[WR] == 0) {
            // only kings and one bishop
            return true;
        }
        if (pC[WB] == 1 && pC[BB] == 0 && (pC[BN] == 1 && pC[WN] == 0 || pC[BN] == 0 && pC[WN] == 1) && pC[BR] == 0 && pC[WR] == 0) {
            // only kings and one knight
            return true;
        }
    }
    return false;
}

// uses FNV1 hash algrorithm to hash and store the current board state
void Board::hashBoard () {
    int numBytes = sizeof(Board::Piece);
    unsigned int hash = 2166136261;
    unsigned int prime = 16777619;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            char* data = (char*)(&state_[i][j]);
            for (int b = 0; b < numBytes; b++) {
                hash = hash ^ data[b];
                hash = hash * prime;
            }
        }
    }
    hash = hash % 8192;
    hashes[hash] += 1;
    lastHash = hash;
}

void Board::unhashBoard () {
    Board::hashBoard();
    hashes[lastHash] -= 2;
}

// verifies if the current board state is a checkmate by checking if there
// are any remaining valid moves (moving into check is an invalid move).
// returns true if 'player' is in checkmate
bool Board::isCheckmate (Board::Player player) {
    std::vector <Board::Move> moves;
    getLegalMoves(player, &moves);
    return moves.size() == 0 && Board::isCheck(player, false);
}

// Returns true if 'pl' is in check
bool Board::isCheck (Board::Player pl, bool real) {
    Board::Pos king_pos;
    Piece p;
    for (int i = 1; i < 9; i++) {
        for (int j = 1; j < 9; j++) {
            king_pos = {.x = i, .y = j};
            p = getPiece(king_pos);
            if ((pl == WHITE && p == WK) || (pl == BLACK && p == BK)) {
                goto outOfLoop; // ya love to see it ;)
            }
        }
    }
    outOfLoop:
    // check all knight jumps from king position
    {
        int x_pos[] = {-2,-1,1,2,2,1,-1,-2};
        int y_pos[] = {1,2,2,1,-1,-2,-2,-1};
        for (int i = 0; i < 8; i++) {
            Board::Pos pos = {.x = king_pos.x + x_pos[i], .y = king_pos.y + y_pos[i]};
            if (isInsideBoard(pos)) {
                p = getPiece(pos);
                if ((pl == WHITE && p == BN) || (pl == BLACK && p == WN)) {
                    if (real) printf("check via knight at %d, %d\n", pos.x, pos.y);
                    return true;
                }
            }
        }
    }

    // check two pawn positions
    {
        Board::Pos pos = king_pos;
        pos.x -= 1;
        if (pl == WHITE) {
            pos.y += 1;
        } else {
            pos.y -= 1;
        }
        if (isInsideBoard(pos)) {
            p = getPiece(pos);
            if ((pl == WHITE && p == BP) || (pl == BLACK && p == WP)) {
                if (real) printf("check via pawn at %d, %d\n", pos.x, pos.y);
                return true;
            }
        }
        pos.x += 2;
        if (isInsideBoard(pos)) {
            p = getPiece(pos);
            if ((pl == WHITE && p == BP) || (pl == BLACK && p == WP)) {
                if (real) printf("check via pawn at %d, %d\n", pos.x, pos.y);
                return true;
            }
        }
    }

    // check surrounding tiles for opponent king
    {
        int x_pos[] = {-1,0,1,1,1,0,-1,-1};
        int y_pos[] = {1,1,1,0,-1,-1,-1,0};
        for (int i = 0; i < 8; i++) {
            Board::Pos pos = {.x = king_pos.x + x_pos[i], .y = king_pos.y + y_pos[i]};
            if (isInsideBoard(pos)) {
                p = getPiece(pos);
                if ((pl == WHITE && p == BK) || (pl == BLACK && p == WK)) {
                    if (real) printf("check via king at %d, %d\n", pos.x, pos.y);
                    return true;
                }
            }
        }
    }
    
    // check in a straight lines
    for (int i = 1; i < 8; i++) {
        Board::Pos pos[] = {king_pos, king_pos, king_pos, king_pos};
        pos[0].x -= i;
        pos[1].x += i;
        pos[2].y -= i;
        pos[3].y += i;

        for (int j = 0; j < 4; j++) {
            if (isInsideBoard(pos[j])) {
                p = getPiece(pos[j]);
                if ((pl == BLACK && (p >= WK && p <= WP)) || (pl == WHITE && (p >= BK && p <= BP))) {
                    // piece on the tile is an enemy
                    if (!isPieceBetween(pos[j], king_pos)) {
                        if ((pl == WHITE && p == BR) || (pl == BLACK && p == WR)) {
                            if (real) printf("check via rook at %d, %d\n", pos[j].x, pos[j].y);
                            return true;
                        }
                        if ((pl == WHITE && p == BQ) || (pl == BLACK && p == WQ)) {
                            if (real) printf("check via queen at %d, %d\n", pos[j].x, pos[j].y);
                            return true;
                        }
                    }
                }
            }
        }
    }

    // check in diagonal lines
    for (int i = 1; i < 8; i++) {
        Board::Pos pos[] = {king_pos, king_pos, king_pos, king_pos};
        pos[0].x -= i; pos[0].y -= i;
        pos[1].x += i; pos[1].y -= i;
        pos[2].x -= i; pos[2].y += i;
        pos[3].x += i; pos[3].y += i;

        for (int j = 0; j < 4; j++) {
            if (isInsideBoard(pos[j])) {
                p = getPiece(pos[j]);
                if ((pl == BLACK && (p >= WK && p <= WP)) || (pl == WHITE && (p >= BK && p <= BP))) {
                    // piece on the tile is an enemy
                    if (!isPieceBetween(pos[j], king_pos)) {
                        if ((pl == WHITE && p == BB) || (pl == BLACK && p == WB)) {
                            if (real) printf("check via bishop at %d, %d\n", pos[j].x, pos[j].y);
                            return true;
                        }
                        if ((pl == WHITE && p == BQ) || (pl == BLACK && p == WQ)) {
                            if (real) printf("check via queen at %d, %d\n", pos[j].x, pos[j].y);
                            return true;
                        }
                    }
                }
            }
        }
    }

    // king not under attack by pawns, knights, or any pieces on a straight or diagonal line
    return false;
}

bool Board::isInsideBoard(Board::Pos pos) {
    if (pos.x < 1 || pos.x > 8) {
        return false;
    } else if (pos.y < 1 || pos.y > 8) {
        return false;
    }
    return true;
}

void Board::getPreviousBoardStates (std::vector <std::vector <std::vector <Board::Piece>>>* states) {
    *states = states_;
}

void Board::getNextBoardStates (std::vector <std::vector <std::vector <Board::Piece>>>* nextStates) {
    std::vector<Board::Move> legalMoves;
    getLegalMoves(turn_, &legalMoves);

    for (Board::Move m : legalMoves) {
        doMove(m);
        nextStates->push_back(state_);
        undoMove();
    }
}

// checks to see if there are pieces in between two points. Points specified
// must be a straight or exact diagonal line, and must lie within the bounds
// of the board
bool Board::isPieceBetween(Board::Pos a, Board::Pos b) {
    int dx = b.x - a.x;
    int dy = b.y - a.y;

    int x_sign = 1;
    int y_sign = 1;
    if (dx < 0) {
        x_sign = -1;
    } 
    if (dy < 0) {
        y_sign = -1;
    }

    Board::Pos tmp = a;

    if ((abs(dx) <= 1) && (abs(dy) <= 1)) {
        return false;
    }
    
    if (dx == 0) {
        // piece is moving along y axis
        for (int i = 1; i < abs(dy); i++) {
            tmp = a;
            tmp.y += i * y_sign;
            if (getPiece(tmp) != EM) {
                return true;
            }
        }
        return false;
    }
    if (dy == 0) {
        // piece is moving along x axis
        for (int i = 1; i < abs(dx); i++) {
            tmp = a;
            tmp.x += i * x_sign;
            if (getPiece(tmp) != EM) {
                return true;
            }
        }
        return false;
    }
    for (int i = 1; i < abs(dx); i++) {
        // piece is moving in a diagonal
        tmp = a;
        tmp.x += i * x_sign;
        tmp.y += i * y_sign;
        if (getPiece(tmp) != EM) {
            return true;
        }
        
    }
    return false;
}

// checks to see if the selected piece is in the specified starting position,
// whether the selected piece is an actual piece, if the start and end positions
// are within the range of the board, if the move is trying to capture a piece
// of the same colour, if the shape of the move is correct for that piece,
// and if the move will leave the player in check or not
bool Board::isLegalMove (Board::Move move, bool real) {
    if (getPiece(move.startPos) != move.piece) {
        return false;
    }
    if (move.piece == EM) {
        return false;
    }
    if (move.startPos.x < 1 || move.startPos.x > 8) {
        return false;
    }
    if (move.startPos.y < 1 || move.startPos.y > 8) {
        return false;
    }
    if (move.endPos.x < 1 || move.endPos.x > 8) {
        return false;
    }
    if (move.endPos.y < 1 || move.endPos.y > 8) {
        return false;
    }
    

    int dx = move.endPos.x - move.startPos.x;
    int dy = move.endPos.y - move.startPos.y;

    if (dx == 0 && dy == 0) {
        if (real) fprintf(stderr, "Invalid move: Movement must be non-zero\n");
        return false;
    }
    
    if ((move.piece >= BK && move.piece <= BP) && (move.taken >= BK && move.taken <= BP)) {
        if (real) fprintf(stderr, "Invalid move: You cannot capture your own piece\n");
        return false;
    }
    if ((move.piece >= WK && move.piece <= WP) && (move.taken >= WK && move.taken <= WP)) {
        if (real) fprintf(stderr, "Invalid move: You cannot capture your own piece\n");
        return false;
    }


    if (move.piece == BK || move.piece == WK) {
        // king is moving
        if (abs(dx) == 2) {
            // must attempt a castle
            // king must have never moved, rook must have never moved
            for (Board::Move prevMove : moves_) {
                if (prevMove.piece == move.piece) {
                    if (real) fprintf(stderr, "Invalid move: King has moved before\n");
                    return false;
                }
                if (prevMove.piece == BR) {
                    if (dx > 0 && prevMove.startPos.x == 8 && prevMove.startPos.y == 8) {
                        if (real) fprintf(stderr, "Invalid move: Rook has moved before\n");
                        return false;
                    }
                    if (dx < 0 && prevMove.startPos.x == 1 && prevMove.startPos.y == 8) {
                        if (real) fprintf(stderr, "Invalid move: Rook has moved before\n");
                        return false;
                    }
                }
                if (prevMove.piece == WR) {
                    if (dx > 0 && prevMove.startPos.x == 8 && prevMove.startPos.y == 1) {
                        if (real) fprintf(stderr, "Invalid move: Rook has moved before\n");
                        return false;
                    }
                    if (dx < 0 && prevMove.startPos.x == 1 && prevMove.startPos.y == 1) {
                        if (real) fprintf(stderr, "Invalid move: Rook has moved before\n");
                        return false;
                    }
                }
            }
            // king and rook have never moved
            Board::Pos tmp = move.startPos;
            if (dx > 0) {
                tmp.x = 8;
                if (isPieceBetween(move.startPos, tmp)) {
                    if (real) fprintf(stderr, "Invalid move: There are pieces between king and rook\n");
                    return false;
                }
            } else if (dx < 0) {
                tmp.x = 1;
                if (isPieceBetween(move.startPos, tmp)) {
                    if (real) fprintf(stderr, "Invalid move: There are pieces between king and rook\n");
                    return false;
                }
            }
        } else if (abs(dx) > 1 || abs(dy) > 1) {
            if (real) fprintf(stderr, "Invalid move: King can only move distance 1\n");
            return false;
        }

        // moving within distance 1 of startPos or was a valid castle
    }

    if (move.piece == BQ || move.piece == WQ) {
        // queen is moving
        if (dx != 0 && dy != 0) {
            // if it is not moving in a straight line
            if (abs(dx) != abs(dy)) {
                if (real) fprintf(stderr, "Invalid move: Queen is not moving in a straight or diagonal line\n");
                return false;
            }
        }
        if (isPieceBetween(move.startPos, move.endPos)) {
            if (real) fprintf(stderr, "Invalid move: There is a piece in the way of the line\n");
            return false;
        }

        // moving in a straight line or diagonal, no pieces in the
        // way
    }

    if (move.piece == BB || move.piece == WB) {
        // bishop is moving
        if (abs(dx) != abs(dy)) {
            if (real) fprintf(stderr, "Invalid move: Rook is not moving in a diagonal line\n");
            return false;
        }
        if (isPieceBetween(move.startPos, move.endPos)) {
            if (real) fprintf(stderr, "Invalid move: There is a piece in the way of the line\n");
            return false;
        }

        // moving in a diagonal, no pieces in the way
    }

    if (move.piece == BN || move.piece == WN) {
        // knight is moving
        if (!(abs(dx) == 1 && abs(dy) == 2)) {
            if (!(abs(dy) == 1 && abs(dx) == 2)) {
                if (real) fprintf(stderr, "Invalid move: Knight is not moving in L shape\n");
                return false;
            }
        }

        // moving in L shape
    }

    if (move.piece == BR || move.piece == WR) {
        // rook is moving
        if (dx != 0) {
            // if moving in x direction
            if (dy != 0) {
                if (real) fprintf(stderr, "Invalid move: Rook is not moving in straight line\n");
                return false;
            }
        }
        if (isPieceBetween(move.startPos, move.endPos)) {
            if (real) fprintf(stderr, "Invalid move: There is a piece in the way of the line\n");
            return false;
        }

        // moving in a straight line, no pieces in the way
    }

    if (move.piece == WP || move.piece == BP) {
        if (!isValidPawnMove(move, real)) {
            return false;
        }
    }

    // is valid move so far. Do the move and see if the board is in check
    moves_.push_back(move);
    states_.push_back(state_);
    Board::switchPlayer();

    if (move.taken == EM && (move.startPos.x - move.endPos.x) != 0 && (move.piece == BP || move.piece == WP)) {
        // have already verified this is a valid move, there must have been
        // an en passant.
        Board::Pos pos = {.x = move.endPos.x, .y = move.startPos.y};
        Board::setPiece(pos, EM);
    }
    if ((move.piece == BK || move.piece == WK) && abs(move.startPos.x - move.endPos.x) == 2) {
        // have already verified this is a valid move, there must have been
        // a castle
        Board::Pos tmp = move.endPos;
        if ((move.endPos.x - move.startPos.x) > 0) {
            tmp.x += 1;
            Board::setPiece(tmp, EM);
            tmp.x -= 2;
            Board::setPiece(tmp, move.piece == BK ? BR : WR);
        }
        if ((move.endPos.x - move.startPos.x) < 0) {
            tmp.x -= 2;
            Board::setPiece(tmp, EM);
            tmp.x += 3;
            Board::setPiece(tmp, move.piece == BK ? BR : WR);
        }
    }
    Board::setPiece(move.startPos, EM);
    Board::setPiece(move.endPos, move.piece);
    // if the moved piece was a pawn that reached the end of the board
    // upgrade it to a queen    
    if (move.piece == WP && move.endPos.y == 8) {
        Board::setPiece(move.endPos, WQ);
    } else if (move.piece == BP && move.endPos.y == 1) {
        Board::setPiece(move.endPos, BQ);
    }

    bool isValid = true;
    
    Board::Player player;
    if (move.piece >= WK && move.piece <= WP) {
        player = WHITE;
    } else {
        player = BLACK;
    }
    if (Board::isCheck(player, real)) {
        if (real) fprintf(stderr, "Invalid move: Move leaves king in check\n");
        isValid = false;
    }
    Board::hashBoard();
    undoMove();
    return isValid;
}

// checks to see if the current move is by a pawn and is
// a valid move.
bool Board::isValidPawnMove (Board::Move move, bool real) {
    if (!(move.piece == WP || move.piece == BP)) {
        // piece moving is not a pawn
        return false;
    }

    int dx = move.endPos.x - move.startPos.x;
    int dy = move.endPos.y - move.startPos.y;
    
    if (abs(dy) > 2 || abs(dx) > 1) {
        // pawn is moving too far in x or y direction
        if (real) fprintf(stderr, "Invalid move: Pawn moves too far\n");
        return false;
    }
    if (abs(dy) == 2) {
        if (abs(dx) != 0) {
            // pawn cannot move in x direction if moving 2 in y
            if (real) fprintf(stderr, "Invalid move: Pawn moves too far\n");
            return false;
        }

        

        // these two checks are okay to use abs(dy). if the pawn 
        // moves two from the starting position it will have either
        // moved two forward or moved outside the board (which will
        // not have happened if this function is being called)
        if (move.piece == WP && move.startPos.y != 2) {
            // pawn can only move 2 forward from starting position
            if (real) fprintf(stderr, "Invalid move: Pawn can only move 2 forward from starting position\n");
            return false;
        }
        if (move.piece == BP && move.startPos.y != 7) {
            // pawn can only move 2 forward from starting position
            if (real) fprintf(stderr, "Invalid move: Pawn can only move 2 forward from starting position\n");
            return false;
        }
        if (isPieceBetween(move.startPos, move.endPos)) {
            if (real) fprintf(stderr, "Invalid move: There is a piece in the way of the line\n");
            return false;
        }
    }
    if ((move.piece == WP && dy < 1) || (move.piece == BP && dy > -1)) {
        // pawn must move forward
        if (real) fprintf(stderr, "Invalid move: Pawn must move forward\n");
        return false;
    }
    // piece has moved exactly 1 or 2 spaces forward

    if (dx == 0) {
        if (getPiece(move.endPos) != EM || move.taken != EM) {
            // pawn cannot capture by moving directly forward
            if (real) fprintf(stderr, "Invalid move: Pawn cannot capture by moving directly forward\n");
            return false;
        } 
        // pawn is moving forward exactly 1 or 2 spaces with 
        // nothing in the way, and is not trying to capture 
        // anything. This is a valid move
    }
    
    if (abs(dx) == 1) {
        // pawn must capture correctly (to reach this point abs(dy) will equal 1)
        if (getPiece(move.endPos) == EM) {
            // player must attempt en passant
            Board::Pos enPassantPlace;
            enPassantPlace.x = move.endPos.x;
            enPassantPlace.y = move.startPos.y;

            bool validEnPassant;
            Board::Move lastMove;
            if (moves_.size() == 0) {
                validEnPassant = false;
            } else {
                lastMove = moves_.back();
                validEnPassant = true;
            }

            if (validEnPassant) {
            if (abs(lastMove.startPos.y - lastMove.endPos.y) != 2) {
                // if the previous move was not a dy of 2
                if (real) fprintf(stderr, "Invalid move: previous move was not a move forward of size 2\n");
                validEnPassant = false;
            }
            if ((lastMove.endPos.x != enPassantPlace.x) || (lastMove.endPos.y != enPassantPlace.y)) {
                // if the previous move did not land on place of en passant capture
                if (real) fprintf(stderr, "Invalid move: last move did not land on place of en passant capture\n");
                validEnPassant = false;
            }

            if (move.piece == WP) {
                if (getPiece(enPassantPlace) != BP) {
                    // if the taken piece is not pawn of opposite colour
                    if (real) fprintf(stderr, "Invalid move: En passant piece is not pawn of opposite colour\n");
                    validEnPassant = false;
                }
                if (lastMove.piece != BP) {
                    // if the previous move was not opposite colour pawn moving
                    if (real) fprintf(stderr, "Invalid move: Previous move was not opposite colour pawn moving\n");
                    validEnPassant = false;
                }
            } else if (move.piece == BP) {
                if (getPiece(enPassantPlace) != BP) {
                    // if the taken piece is not pawn of opposite colour
                    if (real) fprintf(stderr, "Invalid move: En passant piece is not pawn of opposite colour\n");
                    validEnPassant = false;
                }
                if (lastMove.piece != BP) {
                    // if the previous move was not opposite colour pawn moving
                    if (real) fprintf(stderr, "Invalid move: Previous move was not opposite colour pawn moving\n");
                    validEnPassant = false;
                }
            }
            }
            if (!validEnPassant) {
                if (real) fprintf(stderr, "Invalid move: Pawn cannot move like that\n");
                return false;
            }
            // all checks passed, move was a valid en passant
        } else {
            // capture is not an en passant
            if (getPiece(move.endPos) != move.taken) {
                // piece attempting to be captured is not in the right position
                if (real) fprintf(stderr, "Invalid move: Pawn attempted capture but piece was not in position\n");
                return false;
            }
            // pawn is moving diagonally exactly 1 space, and is moving to capture
            // the correct piece. This is a valid capture
        }
    }

    // pawn moved correctly
    return true;
}

void Board::setPiece (Board::Pos pos, Board::Piece piece) {
    state_ [pos.x - 1][pos.y - 1] = piece;
}

void Board::switchPlayer () {
    if (turn_ == WHITE) {
        turn_ = BLACK;
    } else {
        turn_ = WHITE;
    }
}

void Board::undoMove () {
    if (moves_.size() == 0) {
        // no moves to undo
        return;
    }
    Board::unhashBoard();
    state_ = states_.back();
    moves_.pop_back();
    states_.pop_back();
    Board::switchPlayer();
}

void Board::getState(Board::Piece state [16][16]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            Board::Pos pos = {.x = i + 1, .y = j + 1};
            state[i][j] = getPiece(pos);
        }
    }
}

void Board::getLegalMoves (Board::Player player, std::vector <Board::Move>* moves) {
    typedef struct PiecePos {
        Board::Piece piece;
        Board::Pos pos;
    } PiecePos;
    std::vector <PiecePos> piecePos;

    for (int i = 1; i < 9; i++) {
        for (int j = 1; j < 9; j++) {
            Board::Pos pos = {.x = i, .y = j};
            Piece p = getPiece(pos);
            if ((player == WHITE && p >= WK && p <= WP) || (player == BLACK && p >= BK && p <= BP)){
                PiecePos x = {.piece = p, .pos = pos};
                piecePos.push_back(x);
            }
        }
    }

    // check every possible move for each piece and add only moves that satisfy isLegalMove()
    for (PiecePos pp : piecePos) {
        if (pp.piece == WK || pp.piece == BK) {
            // piece is a king
            int pos_x[] = {-1,0,1,1,1,0,-1,-1, -2, 2};
            int pos_y[] = {1,1,1,0,-1,-1,-1,0, 0, 0};
            for (int i = 0; i < 10; i++) {
                Board::Pos pos = {.x = pp.pos.x + pos_x[i], .y = pp.pos.y + pos_y[i]};
                if (isInsideBoard(pos)) {
                    Board::Piece take = getPiece(pos);
                    Board::Move move = {.piece = pp.piece, .startPos = pp.pos, .endPos = pos, .taken = take};
                    if (isLegalMove(move, false)) {
                        moves->push_back(move);
                    }
                }
            }
        }
        if (pp.piece == WQ || pp.piece == BQ) {
            // piece is a queen
            for (int i = 1; i < 8; i++) {
                Board::Pos pos[] = {pp.pos, pp.pos, pp.pos, pp.pos, pp.pos, pp.pos, pp.pos, pp.pos}; // 8 directions of movement
                pos[0].x += i;
                pos[1].x -= i;
                pos[2].y += i;
                pos[3].y -= i;
                pos[4].x += i; pos[4].y -= i;
                pos[5].x -= i; pos[5].y -= i;
                pos[6].x += i; pos[6].y += i;
                pos[7].x -= i; pos[7].y += i;
                for (int j = 0; j < 8; j++) {
                    // for each direction of movement
                    if (isInsideBoard(pos[j])) {
                        Board::Piece take = getPiece(pos[j]);
                        Board::Move move = {.piece = pp.piece, .startPos = pp.pos, .endPos = pos[j], .taken = take};
                        if (isLegalMove(move, false)) {
                            moves->push_back(move);
                        }
                    }
                }
            }
        }
        if (pp.piece == WB || pp.piece == BB) {
            // piece is a bishop
            for (int i = 1; i < 8; i++) {
                Board::Pos pos[] = {pp.pos, pp.pos, pp.pos, pp.pos}; // 4 directions of movement
                pos[0].x += i; pos[0].y -= i;
                pos[1].x -= i; pos[1].y -= i;
                pos[2].x += i; pos[2].y += i;
                pos[3].x -= i; pos[3].y += i;
                for (int j = 0; j < 4; j++) {
                    // for each direction of movement
                    if (isInsideBoard(pos[j])) {
                        Board::Piece take = getPiece(pos[j]);
                        Board::Move move = {.piece = pp.piece, .startPos = pp.pos, .endPos = pos[j], .taken = take};
                        if (isLegalMove(move, false)) {
                            moves->push_back(move);
                        }
                    }
                }
            }
        }
        if (pp.piece == WN || pp.piece == BN) {
            // piece is a knight
            int pos_x[] = {-2,-1,1,2,2,1,-1,-2};
            int pos_y[] = {1,2,2,1,-1,-2,-2,-1};
            for (int i = 0; i < 8; i++) {
                Board::Pos pos = {.x = pp.pos.x + pos_x[i], .y = pp.pos.y + pos_y[i]};
                if (isInsideBoard(pos)) {
                    Board::Piece take = getPiece(pos);
                    Board::Move move = {.piece = pp.piece, .startPos = pp.pos, .endPos = pos, .taken = take};
                    if (isLegalMove(move, false)) {
                        moves->push_back(move);
                    }
                }
            }
        }
        if (pp.piece == WR || pp.piece == BR) {
            // piece is a rook
            for (int i = 1; i < 8; i++) {
                Board::Pos pos[] = {pp.pos, pp.pos, pp.pos, pp.pos}; // 4 directions of movement
                pos[0].x += i;
                pos[1].x -= i;
                pos[2].y += i;
                pos[3].y -= i;
                for (int j = 0; j < 4; j++) {
                    // for each direction of movement
                    if (isInsideBoard(pos[j])) {
                        Board::Piece take = getPiece(pos[j]);
                        Board::Move move = {.piece = pp.piece, .startPos = pp.pos, .endPos = pos[j], .taken = take};
                        if (isLegalMove(move, false)) {
                            moves->push_back(move);
                        }
                    }
                }
            }
        }
        if (pp.piece == WP || pp.piece == BP) {
            // piece is a pawn
            Board::Move move;
            move.piece = pp.piece;
            move.startPos = pp.pos;

            Board::Pos pos = pp.pos;
            if (pp.piece == WP) {
                pos.y += 1;
            } else {
                pos.y -= 1;
            }
            Board::Piece take;
            if (isInsideBoard(pos)) {
                take = getPiece(pos);
                move.endPos = pos;
                move.taken = take;
                if (isLegalMove(move, false)) {
                    moves->push_back(move);
                }
            }
            if (pp.piece == WP) {
                pos.y += 1;
            } else {
                pos.y -= 1;
            }
            if (isInsideBoard(pos)) {
                take = getPiece(pos);
                move.endPos = pos;
                move.taken = take;
                if (isLegalMove(move, false)) {
                    moves->push_back(move);
                }
            }
            if (pp.piece == WP) {
                pos.y -= 1;
            } else {
                pos.y += 1;
            }
            pos.x += 1;
            if (isInsideBoard(pos)) {
                take = getPiece(pos);
                move.endPos = pos;
                if (take == EM) {
                    if (pp.piece == WP) {
                        take = BP;
                    } else {
                        take = WP;
                    }
                }
                if (isLegalMove(move, false)) {
                    moves->push_back(move);
                }
            }
            pos.x -= 2;
            if (isInsideBoard(pos)) {
                take = getPiece(pos);
                move.endPos = pos;
                if (take == EM) {
                    if (pp.piece == WP) {
                        take = BP;
                    } else {
                        take = WP;
                    }
                }
                if (isLegalMove(move, false)) {
                    moves->push_back(move);
                }
            }
        }
    }
}