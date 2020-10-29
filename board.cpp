#include "board.h"

Board::Board () {
    reset();
}

void Board::reset () {
    turn_ = WHITE;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            state_ [i][j] = starting_board_ [i][j];
        }
    }
}

Board::Piece Board::getPiece (Board::Pos pos) {
    return state_ [pos.x - 1][pos.y - 1];
}

int Board::doMove (Board::Move move) {
    if (move.piece == EM) {
        // if player tries to move an empty piece
        return -1;
    }
    if (move.startPos.x < 1 || move.startPos.x > 8 || move.startPos.y < 1 || move.startPos.y > 8) {
        // if starting position is outside the bounds of the board
        return -1;
    }
    if (turn_ == WHITE && (move.piece <= BP) && (move.piece >= BK)) {
        // if it's white's turn and player tries to move a black piece
        return -1;
    }
    if (turn_ == BLACK && (move.piece <= WP) && (move.piece >= WK)) {
        // if it's black's turn and player tries to move a white piece
        return -1;
    }
    if (getPiece(move.startPos) != move.piece) {
        // if the piece the player is trying to move is not in the starting position
        return -1;
    }

    // player has selected a valid piece. Check to see if the place they're moving it to is valid.
    
    if (move.endPos.x < 1 || move.endPos.x > 8 || move.endPos.y < 1 || move.endPos.y > 8) {
        // if ending position is outside the bounds of the board
        return -1;
    }
    if (getPiece(move.endPos) != move.taken) {
        // if the piece the player is trying to take is not in the ending position
        return -1;
    }
}

void Board::setPiece (Board::Pos pos, Board::Piece piece) {
    state_ [pos.x - 1][pos.y - 1] = piece;
}

void Board::undoMove () {
    if (moves.size() == 0) {
        // no moves to undo
        return;
    }
    Board::Move move = moves.back();
    setPiece(move.startPos, move.piece);
    setPiece(move.endPos, move.taken);
    moves.pop_back();
    
    if (turn_ == WHITE) {
        turn_ = BLACK;
    } else {
        turn_ = WHITE;
    }
}