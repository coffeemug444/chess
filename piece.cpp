#include "piece.hpp"

std::ostream& operator<<(std::ostream& out, const PieceType& p)
{
   switch (p)
   {
   case PAWN:   out << "pawn";   break;
   case ROOK:   out << "rook";   break;
   case KNIGHT: out << "knight"; break;
   case BISHOP: out << "bishop"; break;
   case KING:   out << "king";   break;
   case QUEEN:  out << "queen";  break;
   default: break;
   }
   return out;
}