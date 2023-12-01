#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <ranges>
#include <thread>
#include <random>

#include "piece.hpp"
#include "board.hpp"

#include <iostream>

using namespace std::chrono_literals;

int main() 
{
   
   Board board{};
   std::random_device rd;
   std::mt19937 gen(rd());


   // board.printBoard();
   for (int i = 0; i < 500; i++)
   {
      std::cout << "Game " << i << std::endl;
      board.reset();
      std::set<Move> moves = board.getAllLegalMoves();
      while (moves.size() > 0)
      {
         // std::this_thread::sleep_for(1000ms);
         std::uniform_int_distribution<> distr(0, moves.size() - 1);
         auto it = moves.begin();
         std::advance(it, distr(gen));
         board.doMove(*it);
         moves = board.getAllLegalMoves();
         // board.printBoard();
      }
   }

   std::cout << "Ran to completion" << std::endl;



}




