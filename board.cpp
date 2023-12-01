#include "board.hpp"
#include <vector>
#include <algorithm>

#include <iostream>
#include <string>



bool operator<(const std::optional<PieceType>& l, const std::optional<PieceType>& r)
{
   if (not l.has_value() and r.has_value()) return true;
   if (l.has_value() and not r.has_value()) return false;
   if (not l.has_value() and not r.has_value()) return false;
   // both l and r have values
   return static_cast<int>(l.value()) < static_cast<int>(r.value());
}

std::ostream& operator<<(std::ostream& out, const Square& sq)
{
   switch (sq.row)
   {
      case 0: out << "A"; break;
      case 1: out << "B"; break;
      case 2: out << "C"; break;
      case 3: out << "D"; break;
      case 4: out << "E"; break;
      case 5: out << "F"; break;
      case 6: out << "G"; break;
      case 7: out << "H"; break;
      default: return out;
   }
   out << sq.col + 1;
   return out;
}


std::pair<int, int> getDirectionOffset(Direction dir)
{
   switch (dir)
   {
      case N:   return { 1, 0};
      case NE:  return { 1, 1};
      case E:   return { 0, 1};
      case SE:  return {-1, 1};
      case S:   return {-1, 0};
      case SW:  return {-1,-1};
      case W:   return { 0,-1};
      case NW:  return { 1,-1};
      case NNE: return { 2, 1};
      case NEE: return { 1, 2};
      case SEE: return {-1, 2};
      case SSE: return {-2, 1};
      case SSW: return {-2,-1};
      case SWW: return {-1,-2};
      case NWW: return { 1,-2};
      case NNW: return { 2,-1};
      default:  return { 0, 0}; // unreachable
   }
}


void Board::printBoard() const
{
   for (int row = 7; row >= 0; row--)
   {
      std::cout << char('A' + row) << ' ';
      for (int col = 0; col < 8; col++)
      {
         bool inverted = (row&1)^(col&1);
         std::cout << "\033[" << (inverted ? "7m" : "27m");
         OptionalPiece p = getPiece({row, col});
         if (p.has_value())
         {
            bool w = p.value().color == WHITE;
            w = w ^ inverted;
            switch(p.value().type)
            {
               case ROOK:   std::cout << (w?"♜":"♖"); break;
               case KNIGHT: std::cout << (w?"♞":"♘"); break;
               case BISHOP: std::cout << (w?"♝":"♗"); break;
               case QUEEN:  std::cout << (w?"♛":"♕"); break;
               case KING:   std::cout << (w?"♚":"♔"); break;
               case PAWN:   std::cout << (w?"♟":"♙"); break;
            }
            std::cout << ' ';
         }
         else std::cout << "  ";
         std::cout << "\033[27m";
      }
      std::cout << '\n';
   }
   std::cout << "  1 2 3 4 5 6 7 8\n";
}

void Board::doMove(const Move& move)
{
   Piece piece = getPiece(move.start).value();

   bool capture = getPiece(move.end).has_value();
   bool en_passant = piece.type == PAWN and move.end == m_en_passant_square;
   if (en_passant)
   {
      // find and remove the captured pawn
      getPiece(Square{.row = (piece.color == WHITE ? 4 : 3), .col = move.end.col}).reset();
   }

   // castling
   if (piece.type == KING and std::abs(move.end.col - move.start.col) == 2)
   {
      bool kings_side = (move.end.col > move.start.col);
      OptionalPiece& rook = getPiece(Square{.row = move.start.row, .col = (kings_side ? 7 : 0)});
      getPiece(Square{.row = move.start.row, .col = (kings_side ? 5 : 3)}) = rook.value();
      rook.reset();
   }

   // wtf?!?! you can't take the king??!?
   if (getPiece(move.end).has_value() && getPiece(move.end).value().type == KING) 
   {
      std::cout << "uhhhhh tried to take the king??\n";
      throw -1;
   }

   getPiece(move.end) = piece;
   getPiece(move.start).reset();

   if (move.promotion.has_value())
   {
      getPiece(move.end).value().type = move.promotion.value();
   }

   if (piece.type == KING)
   {
      if (piece.color == WHITE)
      {
         m_white_king = move.end;
         m_white_kingside_available = false;
         m_white_queenside_available = false;
      }
      else
      {
         m_black_king = move.end;
         m_black_kingside_available = false;
         m_black_queenside_available = false;
      }
   }

   if (piece.type == ROOK)
   {
      if (piece.color == WHITE)
      {
         if (move.start == Square{.row = 0, .col = 7}) m_white_kingside_available = false;
         if (move.start == Square{.row = 0, .col = 0}) m_white_queenside_available = false;
      }
      else
      {
         if (move.start == Square{.row = 7, .col = 7}) m_black_kingside_available = false;
         if (move.start == Square{.row = 7, .col = 0}) m_black_queenside_available = false;
      }
   }

   m_en_passant_square.reset();
   if (piece.type == PAWN && std::abs(move.start.row - move.end.row) == 2)
   {
      // pawn did a double advance
      int row = (move.start.row + move.end.row) / 2;
      m_en_passant_square = Square{.row = row, .col = move.start.col};
   }

   m_current_player = (piece.color == WHITE ? BLACK : WHITE);
   if (piece.type == PAWN or capture) m_halfmoves = 0;
   else m_halfmoves++;

   if (m_current_player == WHITE) m_fullmoves++;
   rebuildPins(BLACK);
   rebuildPins(WHITE);
   rebuildThreatsAndChecks();

}

std::set<Move> Board::getAllLegalMoves() const
{
   std::set<Move> legal_moves;

   if (m_halfmoves >= 50)
   {
      std::cout << "50 move rule - game over\n";
      return legal_moves;
   }

   // other checks, insufficient material, board repetition



   auto add_legal_moves = [&legal_moves](const std::set<Move>& new_legal_moves) {
      legal_moves.insert(new_legal_moves.begin(), new_legal_moves.end());
   };

   for (int row = 0; row < 8; row++)
   {
      for (int col = 0; col < 8; col++)
      {
         Square square = { .row = row, .col = col };
         const OptionalPiece& p = getPiece(square);
         if (not p.has_value()) continue;
         if (p.value().color != m_current_player) continue;
         switch (p.value().type)
         {
            case PAWN:   add_legal_moves(getAllLegalPawnMoves(square));   break;
            case ROOK:   add_legal_moves(getAllLegalRookMoves(square));   break;
            case KNIGHT: add_legal_moves(getAllLegalKnightMoves(square)); break;
            case BISHOP: add_legal_moves(getAllLegalBishopMoves(square)); break;
            case KING:   add_legal_moves(getAllLegalKingMoves(square));   break;
            case QUEEN:  add_legal_moves(getAllLegalQueenMoves(square));  break;
         }
      }
   }

   if (legal_moves.size() == 0)
   {
      printBoard();
      if (m_check_locations.size() == 0)
      {
         std::cout << "Stalemate\n";
      }
      else
      {
         std::cout << "Checkmate, " << (m_current_player == WHITE ? "black" : "white") << " wins\n";
      }
   }

   return legal_moves;
}

std::set<Move> Board::getAllLegalPawnMoves(const Square& start) const
{
   // if the king is in check from two (or more??) locations
   // then only the king has legal moves
   if (m_check_locations.size() > 1) return {}; 

   std::set<Move> legal_moves;
   const Piece& pawn = getPiece(start).value();
   int dy = pawn.color == WHITE ? 1 : -1;

   Square dst = start;
   dst.row += dy;
   
   if (squareIsOnBoard(dst) and not getPiece(dst).has_value())
   {
      // the space in front of the pawn is free
      bool last_row = dst.row == (pawn.color == WHITE ? 7 : 0);
      if (last_row)
      {
         for (PieceType promotion_type : {ROOK, KNIGHT, BISHOP, QUEEN})
         {
            legal_moves.insert({start, dst, promotion_type});
         }
      }
      else legal_moves.insert({start, dst});
   }

   // if we're on the starting row it could be possible to do a double advance
   if (start.row == (pawn.color == WHITE ? 1 : 6))
   {
      dst.row += dy;
      if (not getPiece(dst).has_value() and not getPiece({.row = dst.row - dy, .col = dst.col}).has_value())
      {
         // the two spaces in front of the pawn are free
         legal_moves.insert({start, dst});
      }
      dst.row -= dy;
   }


   // if the destination square has a piece of the opposite color, or it's an en passant target
   auto legal_pawn_capture = [&](Square dst) {
   return (squareIsOnBoard(dst) and 
           getPiece(dst).has_value() and 
           getPiece(dst).value().color != pawn.color) || 
          (m_en_passant_square.has_value() and 
           m_en_passant_square.value() == dst);
   };

   dst.col += 1;
   if (legal_pawn_capture(dst)) legal_moves.insert({start, dst});

   dst.col -= 2;
   if (legal_pawn_capture(dst)) legal_moves.insert({start, dst});

   // if there are move restrictions (piece is pinned, king is in
   // check), limit moves to the union of legal_moves and restrictions
   return restrictedSubset(legal_moves, moveRestrictions(start));
}

std::set<Move> Board::getAllLegalKnightMoves(const Square& start) const
{
   // if the king is in check from two (or more??) locations
   // then only the king has legal moves
   if (m_check_locations.size() > 1) return {}; 
   std::set<Move> legal_moves;
   const Piece& knight = getPiece(start).value();

   for (Direction dir : knightDirs())
   {
      auto [north, east] = getDirectionOffset(dir);
      Square dst {.row = start.row + north, .col = start.col + east};
      if (squareIsOnBoard(dst) and 
          (not getPiece(dst).has_value() or 
           (getPiece(dst).has_value() and getPiece(dst).value().color != knight.color)))
      {
         legal_moves.insert({start, dst});
      }
   }
   
   return restrictedSubset(legal_moves, moveRestrictions(start));
}

std::set<Move> Board::getAllLegalRookMoves(const Square& start) const
{
   // if the king is in check from two (or more??) locations
   // then only the king has legal moves
   if (m_check_locations.size() > 1) return {}; 
   std::set<Move> legal_moves;
   const Piece& rook = getPiece(start).value();

   for (Direction dir : rankAndFileDirs())
   {
      auto [north, east] = getDirectionOffset(dir);
      Square dst {.row = start.row + north, .col = start.col + east};
      while (squareIsOnBoard(dst))
      {
         if (getPiece(dst).has_value())
         {
            if (getPiece(dst).value().color != rook.color) legal_moves.insert({start, dst});
            break;
         }
         legal_moves.insert({start, dst});
         dst.row += north;
         dst.col += east;
      }
   }
   
   return restrictedSubset(legal_moves, moveRestrictions(start));
}

std::set<Move> Board::getAllLegalBishopMoves(const Square& start) const
{
   // if the king is in check from two (or more??) locations
   // then only the king has legal moves
   if (m_check_locations.size() > 1) return {}; 
   std::set<Move> legal_moves;
   const Piece& bishop = getPiece(start).value();

   for (Direction dir : diagonalDirs())
   {
      auto [north, east] = getDirectionOffset(dir);
      Square dst {.row = start.row + north, .col = start.col + east};
      while (squareIsOnBoard(dst))
      {
         if (getPiece(dst).has_value())
         {
            if (getPiece(dst).value().color != bishop.color) legal_moves.insert({start, dst});
            break;
         }
         legal_moves.insert({start, dst});
         dst.row += north;
         dst.col += east;
      }
   }
   

   return restrictedSubset(legal_moves, moveRestrictions(start));
}

std::set<Move> Board::getAllLegalQueenMoves(const Square& start) const
{
   // if the king is in check from two (or more??) locations
   // then only the king has legal moves
   if (m_check_locations.size() > 1) return {}; 
   std::set<Move> legal_moves;
   const Piece& queen = getPiece(start).value();

   for (Direction dir : queenDirs())
   {
      auto [north, east] = getDirectionOffset(dir);
      Square dst {.row = start.row + north, .col = start.col + east};
      while (squareIsOnBoard(dst))
      {
         if (getPiece(dst).has_value())
         {
            if (getPiece(dst).value().color != queen.color) legal_moves.insert({start, dst});
            break;
         }
         legal_moves.insert({start, dst});
         dst.row += north;
         dst.col += east;
      }
   }
   

   return restrictedSubset(legal_moves, moveRestrictions(start));
}

std::set<Move> Board::getAllLegalKingMoves(const Square& start) const
{
   std::set<Move> legal_moves;
   const Piece& king = getPiece(start).value();
   Color color = king.color;
   const std::set<Square>& threatened_squares = (color == WHITE ? m_threatened_white_squares : m_threatened_black_squares);

   for (Direction dir : queenDirs())
   {
      auto [north, east] = getDirectionOffset(dir);
      Square dst = {.row = start.row + north, .col = start.col + east};
      if (squareIsOnBoard(dst) && threatened_squares.find(dst) == threatened_squares.end() && // square is on the board and not threatened
         (not getPiece(dst).has_value() ||   // there is no piece on the destination square or...
         (getPiece(dst).has_value() && getPiece(dst).value().color != color)))   // the piece on the destination square is the opposite color
      {
         legal_moves.insert({start, dst});
      }
   }

   if (m_check_locations.size() > 0) return legal_moves;
   int king_row = (color == WHITE ? 0 : 7);
   if (kingSideCastlePossible(color))  legal_moves.insert({start, {.row = king_row, .col = 6}});
   if (queenSideCastlePossible(color)) legal_moves.insert({start, {.row = king_row, .col = 2}});
   return legal_moves;
}

bool Board::kingSideCastlePossible(Color color) const
{
   const std::set<Square>& threatened_squares = (color == WHITE ? m_threatened_white_squares : m_threatened_black_squares);
   int king_row = (color == WHITE ? 0 : 7);
   if (not (color == WHITE ? m_white_kingside_available : m_black_kingside_available)) return false;

   Square between_square_1 {.row = king_row, .col = 5};
   Square between_square_2 {.row = king_row, .col = 6};

   if (getPiece(between_square_1).has_value()) return false;
   if (getPiece(between_square_2).has_value()) return false;

   if (threatened_squares.find(between_square_1) != threatened_squares.end() ||
       threatened_squares.find(between_square_2) != threatened_squares.end()) return false;
   
   return true;
}

bool Board::queenSideCastlePossible(Color color) const
{
   const std::set<Square>& threatened_squares = (color == WHITE ? m_threatened_white_squares : m_threatened_black_squares);
   int king_row = (color == WHITE ? 0 : 7);
   if (not (color == WHITE ? m_white_queenside_available : m_black_queenside_available)) return false;

   Square between_square_1 {.row = king_row, .col = 1};
   Square between_square_2 {.row = king_row, .col = 2};
   Square between_square_3 {.row = king_row, .col = 3};

   if (getPiece(between_square_1).has_value()) return false;
   if (getPiece(between_square_2).has_value()) return false;
   if (getPiece(between_square_3).has_value()) return false;

   if (threatened_squares.find(between_square_1) != threatened_squares.end() ||
       threatened_squares.find(between_square_2) != threatened_squares.end() ||
       threatened_squares.find(between_square_3) != threatened_squares.end()) return false;
   
   return true;
}

// If the piece on `square` is pinned it can only move along
// the pin line. If the king is in check then the piece can
// either block the check or take the piece making check.
std::optional<std::set<Move>> Board::moveRestrictions(const Square& start) const
{
   if (m_check_locations.size() > 1) return std::set<Move> {}; // there are not valid moves to be made
   std::set<Move> pin_restrictions;

   const Piece& piece = getPiece(start).value();

   // add pin squares
   std::map<Square, Square> pins = (piece.color == WHITE ? m_pinned_white_pieces : m_pinned_black_pieces);
   if (pins.find(start) != pins.end())
   {
      std::set<Square> pin_line = getSquaresOnLine(pins.at(start), start);
      for (auto pin_square : pin_line) pin_restrictions.insert({start, pin_square});
   } else if (m_check_locations.size() == 0) return {}; // not pinned, not in check

   // this piece is pinned but its king is not in check
   if (m_check_locations.size() != 1) return pin_restrictions;

   // This piece's king is in check

   std::set<Move> check_restrictions;

   // add checking squares
   Square checking_square = m_check_locations.at(0);
   const Piece& checking_piece = getPiece(checking_square).value();
   switch(checking_piece.type)
   {
      case ROOK:
      case BISHOP:
      case QUEEN:
         break;
      case PAWN:
      case KNIGHT:
      {
         // This piece is a pawn or a knight so its only option is to capture the attacking piece
         Move kill_attacker{start, checking_square};
         if (pin_restrictions.size() == 0 || pin_restrictions.find(kill_attacker) != pin_restrictions.end())
            check_restrictions.insert(kill_attacker);
         return check_restrictions;
      }
      default:
         return check_restrictions;
   }

   std::set<Square> check_line = getSquaresOnLine(checking_square, (piece.color == WHITE ? m_white_king : m_black_king));
   for (auto check_square : check_line) 
   {
      Move block_check{start, check_square};
      if (pin_restrictions.size() == 0 || pin_restrictions.find(block_check) != pin_restrictions.end())
         check_restrictions.insert(block_check);
   }
   return check_restrictions;
}

void Board::reset()
{
   std::vector<PieceType> piece_types {
      ROOK,
      KNIGHT,
      BISHOP,
      QUEEN,
      KING,
      BISHOP,
      KNIGHT,
      ROOK
   };

   for (int row = 0; row < 8; row++)
   {
      for (int col = 0; col < 8; col++)
      {
         OptionalPiece& p = getPiece({.row = row, .col = col});
         switch(row)
         {
            case 0:  p = Piece{piece_types.at(col), WHITE}; break;
            case 1:  p = Piece{PAWN, WHITE}; break;
            case 6:  p = Piece{PAWN, BLACK}; break;
            case 7:  p = Piece{piece_types.at(col), BLACK}; break;
            default: p.reset();
         }
      }
   }

   m_current_player = WHITE;
   m_black_kingside_available = true;
   m_black_queenside_available = true;
   m_white_kingside_available = true;
   m_white_queenside_available = true;
   m_en_passant_square.reset();
   m_halfmoves = 0;
   m_fullmoves = 1;

   m_white_king = { .row = 0, .col = 4 };
   m_black_king = { .row = 7, .col = 4 };

   m_pinned_white_pieces.clear();
   m_pinned_black_pieces.clear();

   m_threatened_black_squares.clear();
   m_threatened_white_squares.clear();
   m_check_locations.clear();

   rebuildThreatsAndChecks();
}


void Board::rebuildPins(Color color)
{
   std::map<Square, Square>& pinned_pieces = (color == WHITE ? m_pinned_white_pieces : m_pinned_black_pieces);

   pinned_pieces.clear();
   for (int row = 0; row < 8; row++)
   {
      for (int col = 0; col < 8; col++)
      {
         Square pinning {row, col};
         const OptionalPiece& p = getPiece(pinning);
         if (not p.has_value()) continue; // no piece on this square
         if (p.value().color == color) continue; // this piece is the same color as the king

         std::optional<Square> pinned_piece = pinnedPiece(pinning);
         if (pinned_piece.has_value())
         {
            pinned_pieces[pinned_piece.value()] = pinning;
         }
      }
   }
}


std::optional<Square> Board::pinnedPiece(const Square& attacking_pos) const
{
   const Piece& p = getPiece(attacking_pos).value();
   Color king_color = (p.color == WHITE ? BLACK : WHITE );
   const Square& king_pos = (king_color == WHITE ? m_white_king: m_black_king);
   int dx = king_pos.col - attacking_pos.col;
   int dy = king_pos.row - attacking_pos.row;
   auto rowCheck  = [](int dx, int dy) -> bool { return dx == 0 || dy == 0; };
   auto diagCheck = [](int dx, int dy) -> bool { return dx*dx == dy*dy; }; // abs(dx) == abs(dy) with less writing
   switch (p.type)
   {
      case BISHOP:
         if (not diagCheck(dx, dy)) return {};
         break;
      case ROOK:
         if (not rowCheck(dx, dy)) return {};
         break;
      case QUEEN:
         if (not (rowCheck(dx, dy) || diagCheck(dx, dy))) return {};
         break;
      default: return {};
   }

   // the piece can move in a way to attack the king, check there is exactly 1
   // piece in the way

   bool blocked = false;
   dx = std::clamp(dx, -1, 1);
   dy = std::clamp(dy, -1, 1);
   Square pos = {.row = attacking_pos.row + dy, .col = attacking_pos.col + dx};
   std::optional<Square> pinnedPos;
   while (pos != king_pos)
   {
      const OptionalPiece& p = getPiece(pos);
      if (p.has_value()) 
      {
         if (blocked) return {}; // there's already another piece blocking, this can't be a pin
         blocked = true;
         pinnedPos = pos;
      }
      pos.col += dx;
      pos.row += dy;
   }
   return pinnedPos;
}

void Board::rebuildThreatsAndChecks()
{
   m_threatened_white_squares.clear();
   m_threatened_black_squares.clear();
   m_check_locations.clear();

   for (int row = 0; row < 8; row++)
   {
      for (int col = 0; col < 8; col++)
      {
         Square square {.row = row, .col = col};   // origin square of attacking piece
         OptionalPiece& p = getPiece(square);
         if (not p.has_value()) continue;          // square has no pieces on it
         std::set<Square> new_threatened_squares = threatenedSquares(square);    // the squares this piece is threatening
         Color p_color = p.value().color;
         std::set<Square>& threatened_squares(p_color == WHITE ? m_threatened_black_squares : m_threatened_white_squares);
         threatened_squares.insert(new_threatened_squares.begin(), new_threatened_squares.end());
         if (p_color != m_current_player and new_threatened_squares.contains((m_current_player == WHITE ? m_white_king : m_black_king)))
         {
            m_check_locations.push_back(square);
         }
      }
   }
}


std::set<Square> Board::threatenedSquares(const Square& attacking) const
{
   std::set<Square> threats;
   const Piece& piece = getPiece(attacking).value();
   const Square& opposite_king = (piece.color == WHITE ? m_black_king : m_white_king);

   if (piece.type == PAWN)
   {
      int dy = piece.color == WHITE ? 1 : -1;
      Square threat = attacking;
      threat.row += dy;
      threat.col += 1;
      if (squareIsOnBoard(threat)) threats.insert(threat);
      threat.col -= 2;
      if (squareIsOnBoard(threat)) threats.insert(threat);
      return threats;
   }

   if (piece.type == KNIGHT)
   {
      for (Direction dir : knightDirs())
      {
         auto [north, east] = getDirectionOffset(dir);
         Square threat = {.row = attacking.row + north, .col = attacking.col + east};
         if (squareIsOnBoard(threat)) threats.insert(threat);
      }
      return threats;
   }

   if (piece.type == KING)
   {
      for (Direction dir : queenDirs())
      {
         auto [north, east] = getDirectionOffset(dir);
         Square threat = {.row = attacking.row + north, .col = attacking.col + east};
         if (squareIsOnBoard(threat)) threats.insert(threat);
      }
      return threats;
   }

   for (Direction dir : queenDirs())
   {
      if (piece.type == ROOK && (dir == NE || dir == SE || dir == SW || dir == NW)) continue;
      if (piece.type == BISHOP && (dir == N || dir == E || dir == S || dir == W)) continue;

      auto [north, east] = getDirectionOffset(dir);
      Square threat = {.row = attacking.row + north, .col = attacking.col + east};
      while (squareIsOnBoard(threat))
      {
         threats.insert(threat);
         if (getPiece(threat).has_value())
         {
            // threats continue through kings, ie. they can't move back
            // to escape a threat, they need to move out of the line of sight
            if (threat != opposite_king) break;
         }
         threat.row += north;
         threat.col += east;
      } 
   }
   return threats;
}

bool Board::squareIsOnBoard(const Square& square)
{
   return (0 <= square.row && square.row < 8) &&
          (0 <= square.col && square.col < 8);
}

std::set<Move> Board::restrictedSubset(const std::set<Move>& moves, const std::optional<std::set<Move>>& restrictions)
{

   if (not restrictions.has_value()) return moves;
   std::set<Move> restricted_subset;
   
   for (const Move& move : moves)
   {
      if (std::any_of(restrictions.value().begin(), restrictions.value().end(), 
         [&move](const Move& restriction) -> bool {
            return move.start == restriction.start && move.end == restriction.end;
         }))
      {
         restricted_subset.insert(move);
      }
   }

   return restricted_subset;
}

std::set<Square> Board::getSquaresOnLine(const Square& start, const Square& end)
{
   std::set<Square> squares;
   int dy = std::clamp(end.row - start.row, -1, 1);
   int dx = std::clamp(end.col - start.col, -1, 1);

   Square square = start;
   while (square != end)
   {
      squares.insert(square);
      square.row += dy;
      square.col += dx;
   } 
   
   return squares;
}
