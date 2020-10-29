#ifndef BOARD_H
#define BOARD_H

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

private:
    Piece starting_board_ [16][16] = { {BR, BN, BB, BQ, BK, BB, BN, BR},
                                       {BP, BP, BP, BP, BP, BP, BP, BP},
                                       {EM, EM, EM, EM, EM, EM, EM, EM},
                                       {EM, EM, EM, EM, EM, EM, EM, EM},
                                       {EM, EM, EM, EM, EM, EM, EM, EM},
                                       {EM, EM, EM, EM, EM, EM, EM, EM},
                                       {WP, WP, WP, WP, WP, WP, WP, WP},
                                       {WR, WN, WB, WQ, WK, WB, WN, WR}
                                     };
    Piece state_ [16][16];  // 16*16 array of the board
    Player turn;            // current players turn


public:
    Board ();
    void getState (Piece state [16][16]);   // copies the current state of the board into the array pointed to
    Piece getPiece (Pos pos); // get piece at pos
    int doMove (Piece piece, Pos startPos, Pos endPos); // attempts to play the input move. Returns 0 on success, -1 on failure (invalid move), 1 if they won the game
    bool isCheckmate ();
    void undoMove ();
    void reset ();
};


#endif