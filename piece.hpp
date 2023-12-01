#pragma once
#include <ostream>

enum PieceType
{
   PAWN,
   ROOK,
   KNIGHT,
   BISHOP,
   KING,
   QUEEN
};

std::ostream& operator<<(std::ostream& out, const PieceType& p);

enum Color
{
   BLACK,
   WHITE
};

struct Piece
{
   PieceType type;
   Color color;
};

