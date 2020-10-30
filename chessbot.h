#include "board.h"

// Base class for ChessBot. inherit this class and redefine these
// functions in your own chess bot class

class ChessBot {
public:
    void init(Board::Player player, Board* board);
    void doMove();
};