#pragma once
#include <array>
#include <optional>
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <ostream>

#include "piece.hpp"

// Board class responsible for pretty much everything. 
// A1 is (0,0), H8 is (7,7). (row, col)

template <std::size_t H, std::size_t W, typename T>
using Array2d = std::array<std::array<T, W>, H>;
using OptionalPiece = std::optional<Piece>;
using BoardArray = Array2d<8,8,OptionalPiece>;

struct Square {
   int row;
   int col;
   bool operator==(const Square& other) const { return row == other.row && col == other.col; }
   bool operator<(const Square& other) const { return (8*row + col) < (8*other.row + other.col); }
   bool operator>(const Square& other) const { if (*this < other) return false; else return other < *this; }
};

std::ostream& operator<<(std::ostream& out, const Square& sq);

bool operator<(const std::optional<PieceType>& l, const std::optional<PieceType>& r);

// castling is notated as the king moving
// two spaces left or right from the starting
// position
struct Move {
   Move(const Square& start, const Square& end):start{start},end{end},promotion{}{}
   Move(const Square& start, const Square& end, const PieceType& promotion):start{start},end{end},promotion{promotion}{}
   Square start;
   Square end;
   std::optional<PieceType> promotion;
   bool operator==(const Move& other) const 
   { 
      return start == other.start && end == other.end && (
         (promotion.has_value() && other.promotion.has_value() && promotion.value() == other.promotion.value()) ||
         (not (promotion.has_value() || promotion.has_value()))
      );
   }
   bool operator<(const Move& other) const 
   { 
      if (start < other.start) return true;
      if (start > other.start) return false;
      if (end < other.end) return true;
      if (end > other.end) return false;
      return promotion < other.promotion;
   }


};

enum Direction {
   N,       // North points from A to H
   NE,
   E,
   SE,
   S,
   SW,
   W,
   NW,

   // AND THE HORSE DIRECTIONS!!
   NNE,
   NEE,
   SEE,
   SSE,
   SSW,
   SWW,
   NWW,
   NNW
};

constexpr std::array<Direction, 4> rankAndFileDirs()
{
   return { N, E, S, W };
}

constexpr std::array<Direction, 4> diagonalDirs()
{
   return { NE, SE, SW, NW };
}

constexpr std::array<Direction, 8> queenDirs()
{
   return { N, NE, E, SE, S, SW, W, NW };
}

constexpr std::array<Direction, 8> knightDirs()
{
   return { NNE, NEE, SEE, SSE, SSW, SWW, NWW, NNW };
}


std::pair<int, int> getDirectionOffset(Direction dir);

class Board
{
public:
   Board() { reset(); }
   void reset();
   void printBoard() const;
   
   std::set<Move> getAllLegalMoves() const;

   // does the move and increments the turn counter etc.
   // `move` is assumed to be legal
   void doMove(const Move& move);

private:
   static bool squareIsOnBoard(const Square& square);

   // returns the intersection of `moves` and `restrictions`
   static std::set<Move> restrictedSubset(const std::set<Move>& moves, const std::optional<std::set<Move>>& restrictions);

   // returns the squares from the start up to but not including end
   static std::set<Square> getSquaresOnLine(const Square& start, const Square& end);

   std::set<Move> getAllLegalPawnMoves(const Square& square) const;
   std::set<Move> getAllLegalRookMoves(const Square& square) const;
   std::set<Move> getAllLegalKnightMoves(const Square& square) const;
   std::set<Move> getAllLegalBishopMoves(const Square& square) const;
   std::set<Move> getAllLegalQueenMoves(const Square& square) const;
   std::set<Move> getAllLegalKingMoves(const Square& square) const;

   // If the piece on `square` is pinned it can only move along
   // the pin line. If the king is in check then the piece can
   // either block the check or take the piece making check.
   // if the king isn't in check and the piece isn't pinned then return none
   std::optional<std::set<Move>> moveRestrictions(const Square& square) const;

   bool kingSideCastlePossible(Color color) const;
   bool queenSideCastlePossible(Color color) const;

   OptionalPiece& getPiece(Square square) { return m_board.at(square.row).at(square.col); }
   const OptionalPiece& getPiece(Square square) const { return m_board.at(square.row).at(square.col); }

   void rebuildPins(Color color);  // pieces that are pinned to `color` king
   std::optional<Square> pinnedPiece(const Square& attacking) const; // piece that `attacker` is pinning

   std::map<Square, Square> m_pinned_white_pieces; // maps pinned white pieces to the black pieces pinning them
   std::map<Square, Square> m_pinned_black_pieces; // maps pinned black pieces to the white pieces pinning them

   void rebuildThreatsAndChecks();
   std::set<Square> threatenedSquares(const Square& attacking) const;

   // threatened squares are squares, if a king were on them, would
   // be check. The piece making the threat does not have to be able
   // to move to the threatened square for it to be a threat (eg. a
   // pawn can threaten squares to its front diagonals even if they
   // have no pieces on them, white can threaten squares that have
   // other white pieces on them, a square can be threatened even if
   // acting on the threat would checkmate the player making it)
   std::set<Square> m_threatened_white_squares;
   std::set<Square> m_threatened_black_squares;

   std::vector<Square> m_check_locations; // what squares are making check on the current colors king?

   Square m_white_king;
   Square m_black_king;

   BoardArray m_board;
   Color m_current_player;
   bool m_black_kingside_available;
   bool m_black_queenside_available;
   bool m_white_kingside_available;
   bool m_white_queenside_available;
   std::optional<Square> m_en_passant_square;   // if a pawn moved 2 spaces this is the en passant target
   int m_halfmoves;           // halfmoves since last capture or pawn advance (for 50 move rule)
   int m_fullmoves;           // full moves since game start
};

