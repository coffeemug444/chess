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
        WHITE,
        EMPTY
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
    Piece starting_board_ [16][16] = { {WR, WP, EM, EM, EM, EM, BP, BR},
                                       {WN, WP, EM, EM, EM, EM, BP, BN},
                                       {WB, WP, EM, EM, EM, EM, BP, BB},
                                       {WQ, WP, EM, EM, EM, EM, BP, BQ},
                                       {WK, WP, EM, EM, EM, EM, BP, BK},
                                       {WB, WP, EM, EM, EM, EM, BP, BB},
                                       {WN, WP, EM, EM, EM, EM, BP, BN},
                                       {WR, WP, EM, EM, EM, EM, BP, BR}
                                     };
    Board::Piece state_ [16][16];  // 16*16 array of the board
    Board::Player turn_;            // current players turn

    std::vector <Move> moves_;

    void setPiece (Board::Pos pos, Board::Piece piece);
    void switchPlayer ();
    bool isLegalMove (Board::Move move);

public:
    Board ();
    void getState (Board::Piece state [16][16]);   // copies the current state of the board into the array pointed to
    Board::Piece getPiece (Board::Pos pos); // get piece at pos
    int doMove (Board::Move move); // attempts to play the input move. Returns 0 on success, -1 on failure (invalid move), 1 if they won the game
    Board::Player isCheckmate ();
    void undoMove ();
    void reset ();
};


#endif