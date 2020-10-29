#include "board.h"
#include <vector>
#include <SFML/Graphics.hpp>
#include "stdlib.h"



void updateSprites(std::vector <sf::Sprite*> &sprites, sf::IntRect pieceRects[12], sf::Texture *spriteTexture, Board *board) {
    for (sf::Sprite *sprite : sprites) {
        free(sprite);
    }
    sprites.clear();
    Board::Piece state[16][16];
    board->getState(state);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (state[i + 1][j + 1] != Board::EM) {
                sf::Sprite* x = (sf::Sprite*)malloc(sizeof(sf::Sprite));
                x->setTexture(*spriteTexture);
                x->setTextureRect(pieceRects[state[i][j]]);
                x->move(i * 45.f, (7 - j)*45.f);
            }
        }
    }

}

int main (int argc, char **argv) {
    sf::RenderWindow window(sf::VideoMode(360, 360), "Chess");
    sf::Texture boardTex;
    boardTex.loadFromFile("assets/board.png");
    sf::Sprite boardSprite(boardTex);
    sf::Texture piecesTex;
    piecesTex.loadFromFile("assets/pieces.png");
    sf::IntRect pieceRects[12];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            sf::IntRect pieceRect(0, 0, 45, 45);
            pieceRect.left = 45 * j;
            pieceRect.top = 45 * i;
            pieceRects[j - i * 6] = pieceRect;
        }
    }
    
    Board* board = new Board();
    std::vector <sf::Sprite*> sprites;
    updateSprites(sprites, pieceRects, &piecesTex, board);

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
        for (sf::Sprite *sprite : sprites) {
            window.draw(*sprite);
        }
        //window.draw(shape);
        window.display();
    }

    return 0;
}