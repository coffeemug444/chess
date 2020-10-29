#include "board.h"
#include "stdio.h"

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
        fprintf(stderr, "Invalid move: An empty position was selected to move");
        return -1;
    }
    if (move.startPos.x < 1 || move.startPos.x > 8 || move.startPos.y < 1 || move.startPos.y > 8) {
        fprintf(stderr, "Invalid move: Selected start position is outside the range of the board [x, y] : {1, 8}");
        return -1;
    }
    if (getPiece(move.startPos) != move.piece) {
        fprintf(stderr, "Invalid move: The piece in the starting position is not the piece selected");
        return -1;
    }
    if (turn_ == WHITE && (move.piece <= BP) && (move.piece >= BK)) {
        fprintf(stderr, "Invalid move: It is white's turn to move, but a black piece was selected");
        return -1;
    }
    if (turn_ == BLACK && (move.piece <= WP) && (move.piece >= WK)) {
        fprintf(stderr, "Invalid move: It is black's turn to move, but a white piece was selected");
        return -1;
    }
    

    // player has selected a valid piece. Check to see if the place they're moving it to is valid.
    
    if (move.endPos.x < 1 || move.endPos.x > 8 || move.endPos.y < 1 || move.endPos.y > 8) {
        fprintf(stderr, "Invalid move: Selected end position is outside the range of the board [x, y] : {1, 8}");
        return -1;
    }
    if (getPiece(move.endPos) != move.taken) {
        fprintf(stderr, "Invalid move: The piece in the ending position is not the piece specified");
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