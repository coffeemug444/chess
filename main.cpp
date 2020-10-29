#include "board.h"
#include <vector>
#include <SFML/Graphics.hpp>
#include "stdio.h"



int main (int argc, char **argv) {
    sf::RenderWindow window(sf::VideoMode(360, 360), "Chess");
    sf::Texture boardTex;
    boardTex.loadFromFile("assets/board.png");
    sf::Sprite boardSprite(boardTex);
    //boardSprite.setTextureRect(sf::IntRect(0,0,360,360));

    
    sf::Texture piecesTex;
    piecesTex.loadFromFile("assets/pieces.png");
    
    sf::IntRect pieceRects[12];
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            sf::IntRect pieceRect(0, 0, 45, 45);
            pieceRect.left = 45 * j;
            pieceRect.top = 45 * i;

            pieceRects[6 + j - 6 * i] = pieceRect;
        }
    }

    
    Board* board = new Board();
    std::vector <sf::Sprite> sprites;
    Board::Piece state[16][16];
    board->getState(state);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (state[i][j] != Board::EM) {
                sprites.push_back(sf::Sprite(piecesTex));
                sprites.back().setTextureRect(pieceRects[state[i][j]]);
                sprites.back().setPosition(45*i, 45*j);
            }
        }
    }

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(boardSprite);
        
        for (sf::Sprite sprite : sprites) {
            window.draw(sprite);
        }
        
        window.display();
    }

    return 0;
}