#include "board.h"
#include "stdio.h"
#include "stdlib.h"

Board::Board () {
    reset();
}

void Board::reset () {
    turn_ = WHITE;
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
    if (turn_ == WHITE && (move.taken <= WP) && (move.taken >= WK)) {
        fprintf(stderr, "Invalid move: You cannot take your own piece\n");
        return -1;
    }
    if (turn_ == BLACK && (move.taken <= BP) && (move.taken >= BK)) {
        fprintf(stderr, "Invalid move: You cannot take your own piece\n");
        return -1;
    }
    if (!Board::isLegalMove(move)) {
        fprintf(stderr, "Invalid move: The specified piece cannot move like that\n");
        return -1;
    }

    moves_.push_back(move);
    states_.push_back(state_);
    
    if (getPiece(move.endPos) == EM) {
        // have already verified this is a valid move, there must have been
        // an en passant.
        Board::Pos pos = {.x = move.endPos.x, .y = move.startPos.y};
        Board::setPiece(pos, EM);
    }
    Board::setPiece(move.startPos, EM);
    Board::setPiece(move.endPos, move.piece);
    if (move.piece == WP && move.endPos.y == 8) {
        Board::setPiece(move.endPos, WQ);
    } else if (move.piece == BP && move.endPos.y == 1) {
        Board::setPiece(move.endPos, BQ);
    }
    
    Board::switchPlayer();
    if (Board::isCheckmate() == turn_) {
        fprintf(stderr, "Invalid move: Cannot move into check or checkmate\n");
        Board::undoMove();
        return -1;
    } else if (Board::isCheckmate() != EMPTY) {
        // last player to move won the game
        return 1;
    }

    // valid move, game is still continuing
    return 0;
}

// verifies if the current board state is a checkmate. Also verifies if
// a king is in check and it is the other player's turn
Board::Player isCheckmate () {

    return Board::EMPTY;
}


// checks to see if there are pieces in between two points. Points specified
// must be a straight or exact diagonal line, and must lie within the bounds
// of the board
bool Board::isPieceBetween(Board::Pos a, Board::Pos b) {
    int dx = a.x - b.x;
    int dy = a.y - b.y;

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

// this function only checks the validity of the movement of a piece.
// it is assumed that the player is not moving into check or checkmate, 
// not taking their own piece, and the starting and ending points are 
// within the bounds of the board. Calls isValidPawnMove() which can
// change the game state
bool Board::isLegalMove (Board::Move move) {
    

    int dx = move.startPos.x - move.endPos.x;
    int dy = move.startPos.y - move.endPos.y;

    if (dx == 0 && dy == 0) {
        // movement must be non-zero
        return false;
    }

    if (move.piece == BK || move.piece == WK) {
        // king is moving
        if (abs(dx) > 1 || abs(dy) > 1) {
            return false;
        }

        // moving within distance 1 of startPos
        return true;
    }

    if (move.piece == BQ || move.piece == WQ) {
        // queen is moving
        if (dx != 0 && dy != 0) {
            // if it is not moving in a straight line
            if (abs(dx) != abs(dy)) {
                // if it is not moving in an exact diagonal
                return false;
            }
        }
        if (isPieceBetween(move.startPos, move.endPos)) {
            return false;
        }

        // moving in a straight line or diagonal, no pieces in the
        // way
        return true;
    }

    if (move.piece == BB || move.piece == WB) {
        // bishop is moving
        if (abs(dx) != abs(dy)) {
            // if it is not moving in an exact diagonal
            return false;
        }
        if (isPieceBetween(move.startPos, move.endPos)) {
            return false;
        }

        // moving in a diagonal, no pieces in the way
        return true;
    }

    if (move.piece == BN || move.piece == WN) {
        // knight is moving
        if (abs(dx) != 1 && abs(dy) != 2) {
            if (abs(dy) != 1 && abs(dx) != 2) {
                // not moving in either L shape
                return false;
            }
        }

        // moving in L shape
        return true;
    }

    if (move.piece == BR || move.piece == WR) {
        // rook is moving
        if (dx != 0) {
            // if moving in x direction
            if (dy != 0) {
                // if also moving in y direction
                return false;
            }
        }

        // moving in a straight line
        return true;
    }

    if (move.piece == WP || move.piece == BP) {
        if (!isValidPawnMove(move)) {
            return false;
        }
        return true;
    }
}

// checks to see if the current move is by a pawn and is
// a valid move.
bool Board::isValidPawnMove (Board::Move move) {
    if (!(move.piece == WP || move.piece == BP)) {
        // piece moving is not a pawn
        return false;
    }

    int dx = move.startPos.x - move.endPos.x;
    int dy = move.startPos.y - move.endPos.y;
    
    if (abs(dy) > 2 || abs(dx) > 1) {
        // pawn is moving too far in x or y direction
        return false;
    }
    if (abs(dy) == 2) {
        if (abs(dx) != 0) {
            // pawn cannot move in x direction if moving 2 in y
            return false;
        }

        if (isPieceBetween(move.startPos, move.endPos)) {
            return false;
        }

        // these two checks are okay to use abs(dy). if the pawn 
        // moves two from the starting position it will have either
        // moved two forward or moved outside the board (which will
        // not have happened if this function is being called)
        if (move.piece == WP && move.startPos.y != 2) {
            // pawn can only move 2 forward from starting position
            return false;
        }
        if (move.piece == BP && move.startPos.y != 7) {
            // pawn can only move 2 forward from starting position
            return false;
        }
    }
    if ((move.piece == WP && dy < 1) || (move.piece == BP && dy > -1)) {
        // pawn must move forward
        return false;
    }
    // piece has moved exactly 1 or 2 spaces forward

    if (dx == 0) {
        if (getPiece(move.endPos) != EM || move.taken != EM) {
            // pawn cannot capture by moving directly forward
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

            Board::Move lastMove = moves_.back();

            if (abs(lastMove.startPos.y - lastMove.endPos.y) != 2) {
                // if the previous move was not a dy of 2
                return false;
            }
            if ((lastMove.endPos.x != enPassantPlace.x) || (lastMove.endPos.y != enPassantPlace.y)) {
                // if the previous move did not land on place of en passant capture
                return false;
            }

            if (move.piece == WP) {
                if (move.taken != BP) {
                    // if the taken piece is not pawn of opposite colour
                    return false;
                }
                if (lastMove.piece != BP) {
                    // if the previous move was not opposite colour pawn moving
                    return false;
                }
            } else if (move.piece == BP) {
                if (move.taken != BP) {
                    // if the taken piece is not pawn of opposite colour
                    return false;
                }
                if (lastMove.piece != BP) {
                    // if the previous move was not opposite colour pawn moving
                    return false;
                }
            }
            // all checks passed, move was a valid en passant
        } else {
            // capture is not an en passant
            if (getPiece(move.endPos) != move.taken) {
                // piece attempting to be captured is not in the right position
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
    state_ = states_.back();
    moves_.pop_back();
    states_.pop_back();

    Board::switchPlayer();
}

void Board::getState(Board::Piece state [16][16]) {
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            Board::Pos pos = {.x = i + 1, .y = j + 1};
            state[i][j] = getPiece(pos);
        }
    }
}