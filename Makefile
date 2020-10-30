main.out:	board.cpp main.cpp
	g++ -g -o main.out board.cpp main.cpp -lsfml-graphics -lsfml-window -lsfml-system
