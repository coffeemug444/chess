#include "board.h"
#include "stdio.h"
#include "stdlib.h"

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
    if (getPiece(move.endPos) != move.taken) {
        fprintf(stderr, "Invalid move: The piece in the ending position is not the piece specified\n");
        return -1;
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
    Board::setPiece(move.startPos, EM);
    Board::setPiece(move.endPos, move.piece);
    Board::switchPlayer();
    if (Board::isCheckmate() == turn_) {
        fprintf(stderr, "Invalid move: Cannot move into checkmate\n");
        Board::undoMove();
        return -1;
    } else if (Board::isCheckmate() != EMPTY)
    {
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
// within the bounds of the board.
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
        return true;
    }

    if (move.piece == BQ || move.piece == WQ) {
        // queen is moving
        if (dx == 0) {

        }
    }
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
    Board::Move move = moves_.back();
    setPiece(move.startPos, move.piece);
    setPiece(move.endPos, move.taken);
    moves_.pop_back();

    Board::switchPlayer();
}