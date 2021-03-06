#include "board.h"
#include <vector>
#include <SFML/Graphics.hpp>
#include "stdio.h"

void updateSprites(std::vector <sf::Sprite> *sprites, sf::Texture *piecesTex, sf::IntRect pieceRects[12], Board *board) {
    sprites->clear();
    Board::Piece state[16][16];
    board->getState(state);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (state[i][j] != Board::EM) {
                sprites->push_back(sf::Sprite(*piecesTex));
                sprites->back().setTextureRect(pieceRects[state[i][j]]);
                sprites->back().setPosition(45*i, 45*(7 - j));
            }
        }
    }
}

typedef struct SelectedPiece {
    bool selected;
    Board::Piece piece;
    Board::Pos pos;
} SelectedPiece;

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
    updateSprites(&sprites, &piecesTex, pieceRects, board);
    

    SelectedPiece selectedPiece = {.selected = false};
    sf::RectangleShape selection;
    selection.setSize(sf::Vector2f(45.f, 45.f));
    selection.setFillColor(sf::Color(0x7dff7d64)); // a nice transparent green
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::MouseButtonPressed) {
                sf::Mouse::Button button = event.mouseButton.button;
                if (button == sf::Mouse::Button::Right) {
                    selectedPiece.selected = false;
                }
                if (button == sf::Mouse::Button::Left) {
                    Board::Pos pos;
                    pos.x = 1 + (event.mouseButton.x / 45);
                    pos.y = 8 - (event.mouseButton.y / 45);
                    Board::Piece piece = board->getPiece(pos);
                    if (selectedPiece.selected) {
                        Board::Move move = {.piece = selectedPiece.piece, .startPos = selectedPiece.pos, .endPos = pos, .taken = piece};
                        selectedPiece.selected = false;
                        board->doMove(move);
                        updateSprites(&sprites, &piecesTex, pieceRects, board);
                    } else {
                        selectedPiece.selected = true;
                        selectedPiece.piece = piece;
                        selectedPiece.pos = pos;
                        selection.setPosition(45.f * (pos.x - 1), 45.f * (8 - pos.y));
                    }
                }
            }
            if (event.type == sf::Event::KeyPressed) {
                sf::Keyboard::Key key = event.key.code;
                if (key == sf::Keyboard::R) {
                    // restart
                    board->reset();
                    updateSprites(&sprites, &piecesTex, pieceRects, board);
                }
                if (key == sf::Keyboard::Z) {
                    // undo move
                    board->undoMove();
                    updateSprites(&sprites, &piecesTex, pieceRects, board);
                }
            }
        }

        window.clear();
        
        window.draw(boardSprite);

        for (sf::Sprite sprite : sprites) {
            window.draw(sprite);
        }
        
        if (selectedPiece.selected) {
            window.draw(selection);
        }
        window.display();
    }

    return 0;
}