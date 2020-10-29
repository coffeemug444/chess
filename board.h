#ifndef BOARD_H
#define BOARD_H

#include <vector>

class Board {
public:
    // type declarations

    typedef enum Piece {
        BK = 0, // Black King
        BQ,     // Black Queen
        BB,     // Black Bishop
        BN,     // Black Knight
        BR,     // Black Rook
        BP,     // Black Pawn

        WK,     // White King
        WQ,     // White Queen
        WB,     // White Bishop
        WN,     // White Knight
        WR,     // White Rook
        WP,     // White Pawn

        EM      // Empty
    } Piece;

    typedef enum Player {
        BLACK = 0,
        WHITE
    } Player;

    typedef struct Pos {
        int x;
        int y;
    } Pos;

    typedef struct Move {
        Board::Piece piece;
        Board::Pos startPos;
        Board::Pos endPos;
        Board::Piece taken;
    } Move;

private:
    std::vector <std::vector <Board::Piece>> state_;  // 16*16 array of the board
    Board::Player turn_;            // current players turn

    std::vector <Move> moves_;
    std::vector <std::vector <std::vector <Board::Piece>>> states_; // vector of all previous board states

    void setPiece (Board::Pos pos, Board::Piece piece);
    void switchPlayer ();
    bool isPieceBetween (Board::Pos a, Board::Pos b);
    bool isValidPawnMove (Board::Move move, bool real);
    bool isInsideBoard (Board::Pos pos);

public:
    Board ();
    void getState (Board::Piece state [16][16]);   // copies the current state of the board into the array pointed to
    Board::Piece getPiece (Board::Pos pos); // get piece at pos where [x, y] : {1, 8}
    int doMove (Board::Move move); // attempts to play the input move. Returns 0 on success, -1 on failure (invalid move), 1 if they won the game
    bool isCheckmate (Board::Player player);
    bool isCheck (Board::Player player);
    bool isLegalMove (Board::Move move, bool real);
    std::vector<Board::Move> getLegalMoves(Board::Player player);
    void undoMove ();
    void reset ();
};


#endif