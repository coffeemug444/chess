#include "board.h"

Board::Board () {
    reset();
}

void Board::reset () {
    turn = WHITE;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            state_ [i][j] = starting_board_ [i][j];
        }
    }
}

Board::Piece Board::getPiece (Pos pos) {
    return state_ [pos.x][pos.y];
}

int Board::doMove (Piece piece, Pos startPos, Pos endPos) {
    if (piece == EM) {
        // if player tries to move an empty piece
        return -1;
    }
    if (turn == WHITE && piece <= BP) {
        // if it's white's turn and player tries to move a black piece
        return -1;
    }
    if (turn == BLACK && piece >= WK) {
        // if it's black's turn and player tries to move a white piece
        return -1;
    }
    if (getPiece(startPos) != piece) {
        // if the piece the player is trying to move is not in the starting position
        return -1;
    }

    // player has selected a valid piece. Check to see if the place they're moving it to is valid.
}